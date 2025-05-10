import time
import torch
import torch.nn as nn
import torchvision
import argparse
import copy
import torchvision.transforms as transforms
from typing import Optional
from utils import read_ints_binary
import struct



def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./datas', help='root directory of imagenet dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=768, help='batch size for inference')
    parser.add_argument('--model', type=str, default='vit', help='Selected model, current support vit and resnet', required=True)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    parser.add_argument('--fast', type=bool, default=False, help='Fast mode')
    parser.add_argument('--phy', type=str, help='Physical bit flip index sequence')
    return parser.parse_args()


def prepare_dataset_imagenet(root_dir, batch_size):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.ImageNet(root=root_dir, split='val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=128, pin_memory=True)
    return testloader


def prepare_model(modelname: str):
    if modelname == "vit":
        return torchvision.models.vit_b_16(pretrained=True), torchvision.models.vit_b_16(pretrained=True)
    else:
        return torchvision.models.resnet50(pretrained=True), torchvision.models.resnet50(pretrained=True)

def flip(base_parameters, name, count):
    p = base_parameters
    tensor = p[name]
    if tensor.dtype == torch.float32:
        bits_per_elem = 32
        pack_format, unpack_format = 'f', 'I'
    elif tensor.dtype == torch.float64:
        bits_per_elem = 64
        pack_format, unpack_format = 'd', 'Q'
    elif tensor.dtype == torch.int32:
        bits_per_elem = 32
        pack_format, unpack_format = 'i', 'I'
    elif tensor.dtype == torch.int64:
        bits_per_elem = 64
        pack_format, unpack_format = 'q', 'Q'
    else:
        raise TypeError(f"Unsupported dtype {tensor.dtype}")
    numel = tensor.numel()
    total_bits = numel * bits_per_elem
    if not (0 <= count < total_bits):
        raise IndexError(f"Bit index out of range: must be in [0, {total_bits}), got {count}")

    flat_tensor = tensor.reshape(-1).clone()
    elem_idx = count // bits_per_elem
    bit_idx = count % bits_per_elem

    elem_val = flat_tensor[elem_idx].item()

    bytes_elem = struct.pack(pack_format, elem_val)
    unsigned_val = int.from_bytes(bytes_elem, byteorder='little')

    unsigned_flipped = unsigned_val ^ (1 << bit_idx)

    flipped_bytes = unsigned_flipped.to_bytes(bits_per_elem // 8, byteorder='little')
    flipped_val = struct.unpack(pack_format, flipped_bytes)[0]

    flat_tensor[elem_idx] = flipped_val

    modified_tensor = flat_tensor.reshape(tensor.shape).to(tensor.device)

    out2 = copy.deepcopy(base_parameters)
    out2[name] = modified_tensor
    return out2



def eval(model: torch.nn.Module, std_model: torch.nn.Module, testloader: torch.utils.data.DataLoader, file_path, phy = None):
    model = model.cuda()
    std_model = std_model.cuda()
    torch.user_save.monitor(model)
    infos = []
    for name, tensor in std_model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.is_cuda:
            addr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            if phy is not None and nbytes < 512 * 512 * 16:
                continue
            infos.append((name, addr, nbytes))
    infos.sort(key=lambda x: x[1])
    MIN_BYTES = 512
    groups = []
    cur_names = []
    cur_start = None
    cur_end   = None
    for name, addr, size in infos:
        if cur_start is None:
            cur_start, cur_end, cur_names = addr, addr + size, [name]
        else:
            if addr == cur_end:
                cur_end = addr + size
                cur_names.append(name)
            else:
                total = cur_end - cur_start
                if total >= MIN_BYTES:
                    groups.append((cur_start, cur_end, total, cur_names))
                cur_start, cur_end, cur_names = addr, addr + size, [name]
    if cur_start is not None:
        total = cur_end - cur_start
        if total >= MIN_BYTES:
            groups.append((cur_start, cur_end, total, cur_names))
    if len(groups) == 0:
        groups.append((infos[0][1], infos[0][1] + infos[0][2], infos[0][2], infos[0][0]))
    if phy is not None:
        range_list = read_ints_binary(phy, 'little', False, 4096)
    else:
        range_list = [i for i in range(4096)]
        
    while len(groups) > 0:
        cur_start, cur_end, total, cur_names = groups[0]
        print(f'[+] Select Range: {(cur_start, cur_end)}, target {cur_names[0]}')
        base_parameters = copy.deepcopy(std_model.state_dict())
        correct = 0
        total = 0
        std_correct = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to('cuda'), labels.to('cuda')
                std_outputs = std_model(images)
                _, std_predicted = torch.max(std_outputs.data, 1)
                total += labels.size(0)
                std_correct += (std_predicted == labels).sum().item()
            print(f'[*] Total correct: {std_correct} / {total}')
        f = open(file_path, 'a')
        f.write(f'0,{std_correct}\n')
        f.close()
        for BITS_CNT in range_list:
            base_parameters = flip(base_parameters, cur_names[0], BITS_CNT)
            model.load_state_dict(base_parameters)
            correct = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = torch.user_save.inference(model(images))
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            print(f"[*] [Flip {BITS_CNT}] Accuracy: {100 * correct / total}%")
            f = open(file_path, 'a')
            f.write(f'{BITS_CNT + 1},{correct}\n')
            f.close()
        groups = groups[1:]



def eval_fast(model: torch.nn.Module, std_model: torch.nn.Module, testloader: torch.utils.data.DataLoader, file_path, phy = None):
    model = model.cuda()
    std_model = std_model.cuda()
    torch.user_save.monitor(model)
    infos = []
    for name, tensor in std_model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.is_cuda:
            addr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            infos.append((name, addr, nbytes))
    infos.sort(key=lambda x: x[1])
    MIN_BYTES = 512
    groups = []
    cur_names = []
    cur_start = None
    cur_end   = None
    for name, addr, size in infos:
        if cur_start is None:
            cur_start, cur_end, cur_names = addr, addr + size, [name]
        else:
            if addr == cur_end:
                cur_end = addr + size
                cur_names.append(name)
            else:
                total = cur_end - cur_start
                if total >= MIN_BYTES:
                    groups.append((cur_start, cur_end, total, cur_names))
                cur_start, cur_end, cur_names = addr, addr + size, [name]
    if cur_start is not None:
        total = cur_end - cur_start
        if total >= MIN_BYTES:
            groups.append((cur_start, cur_end, total, cur_names))
    if len(groups) == 0:
        groups.append((infos[0][1], infos[0][1] + infos[0][2], infos[0][2], infos[0][0]))
    if phy is not None:
        range_list = read_ints_binary(phy, 'little', False, 4096)
    else:
        range_list = [i for i in range(4096)]
        
    while len(groups) > 0:
        cur_start, cur_end, total, cur_names = groups[0]
        print(f'[+] Select Range: {(cur_start, cur_end)}, target {cur_names[0]}')
        base_parameters = copy.deepcopy(std_model.state_dict())
        correct = 0
        total = 0
        std_correct = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to('cuda'), labels.to('cuda')
                std_outputs = std_model(images)
                _, std_predicted = torch.max(std_outputs.data, 1)
                total += labels.size(0)
                std_correct += (std_predicted == labels).sum().item()
            print(f'[*] Total correct: {std_correct} / {total}')
        f = open(file_path, 'a')
        f.write(f'0,{std_correct}\n')
        f.close()
        for idx, BITS_CNT in enumerate(range_list):
            base_parameters = flip(base_parameters, cur_names[0], BITS_CNT)
            model.load_state_dict(base_parameters)
            if idx < 4095:
                continue # for fast detection, one test is ok
            correct = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = torch.user_save.inference(model(images))
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            print(f"[*] [Flip {BITS_CNT}] Accuracy: {100 * correct / total}%")
            f = open(file_path, 'a')
            f.write(f'{BITS_CNT + 1},{correct}\n')
            f.close()
        break # for fast detection, one test is ok



if __name__ == '__main__':
    args = parsing_args()
    testloader = prepare_dataset_imagenet(args.root_dir, args.batch_size)
    model, std_model = prepare_model(args.model)
    if not args.fast:
        eval(model, std_model, testloader, args.output, args.phy)
    else:
        eval_fast(model, std_model, testloader, args.output, args.phy)
    