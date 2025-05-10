import time
import torch
import torch.nn as nn
import torchvision
import argparse
import copy
import torchvision.transforms as transforms
from typing import Optional

from collections import Counter


def average_of_top_95_percent(lst):
    count = Counter(lst)
    
    sorted_numbers = sorted(count.items(), key=lambda x: x[1], reverse=True)
    
    total_count = sum(count.values())
    threshold = 0.95 * total_count
    accumulated_count = 0
    selected_numbers = []
    
    for number, occurrences in sorted_numbers:
        if accumulated_count >= threshold:
            break
        selected_numbers.append(number)
        accumulated_count += occurrences
    if not selected_numbers:
        return 0
    average = sum(selected_numbers) / len(selected_numbers)
    return average



def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./datas', help='root directory of imagenet dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for inference')
    parser.add_argument('--model', type=str, default='vit', help='Selected model, current support vit and resnet')
    parser.add_argument('--output', type=str, help='Output file')
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
    if tensor.dtype == torch.float32 or tensor.dtype == torch.int32:
        bits_per_elem = 32
        int_type = torch.int32
    elif tensor.dtype == torch.float64 or tensor.dtype == torch.int64:
        bits_per_elem = 64
        int_type = torch.int64
    else:
        raise TypeError(f"Unsupported dtype {tensor.dtype}. Only float32/64 or int32/64 are supported.")
    numel = tensor.numel()
    total_bits = numel * bits_per_elem
    if not (0 <= count < total_bits):
        raise IndexError(f"Bit index out of range: must be in [0, {total_bits}), got {count}")
    flat = tensor.contiguous().clone().view(-1)
    as_int = flat.view(int_type)
    elem_idx = count // bits_per_elem
    bit_idx  = count %  bits_per_elem
    mask = (1 << bit_idx)
    mask = torch.tensor(mask, dtype=int_type, device=as_int.device)
    as_int[elem_idx] = as_int[elem_idx] ^ mask
    out = as_int.view(flat.dtype).view(tensor.shape)
    out2 = copy.deepcopy(base_parameters)
    out2[name] = out
    return out2



def eval(model: torch.nn.Module, std_model: torch.nn.Module, testloader: torch.utils.data.DataLoader, args):
    model = model.cuda()
    std_model = std_model.cuda()
    correct = 0
    total = 0
    std_correct = 0
    time_sequence = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        history = []
        for images, labels in testloader:
            start_event.record()
            images, labels = images.to('cuda'), labels.to('cuda')
            std_outputs = std_model(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            _, std_predicted = torch.max(std_outputs.data, 1)
            total += labels.size(0)
            std_correct += (std_predicted == labels).sum().item()
            time_sequence.append((labels.size(0), elapsed_time_ms))
            print(f'[+] Current size: {labels.size(0)}, elapsed time: {elapsed_time_ms}ms')
            if labels.size(0) != args.batch_size:
                continue
            history.append(int(elapsed_time_ms))
            break
        t = average_of_top_95_percent(history)
        print(f'[*] Total correct: {std_correct} / {total}, Time: {t} ms')
        print(f'[*] Calculated AccurateLatency {t * (1 + ((total - std_correct) / total))}')
        baseline_acc = t * (1 + ((total - std_correct) / total))
        total = 0
        history = []
        for images, labels in testloader:
            start_event.record()
            images, labels = images.to('cuda'), labels.to('cuda')
            std_outputs = model(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            _, std_predicted = torch.max(std_outputs.data, 1)
            total += labels.size(0)
            correct += (std_predicted == labels).sum().item()
            time_sequence.append((labels.size(0), elapsed_time_ms))
            print(f'[+] Current size: {labels.size(0)}, elapsed time: {elapsed_time_ms}ms')
            if labels.size(0) != args.batch_size:
                continue
            history.append(int(elapsed_time_ms))
            break
        t = average_of_top_95_percent(history)
        print(f'[*] Total correct for Rednet: {correct} / {total}, Time: {t} ms')
        print(f'[*] Calculated AccurateLatency {t * (1 + ((total - correct) / total))}')
        target_acc = t * (1 + ((total - correct) / total))
        print(f'[*] DrDNA to normalized: {target_acc / baseline_acc}')
        f = open(args.output, 'a')
        f.write(f'1,{target_acc / baseline_acc}\n')
        f.close()



if __name__ == '__main__':
    args = parsing_args()
    testloader = prepare_dataset_imagenet(args.root_dir, args.batch_size)
    model, std_model = prepare_model(args.model)
    eval(model, std_model, testloader, args)