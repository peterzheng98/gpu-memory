.PHONY: virt phy virtFast phyFast
virt:
	$(BASE_PYTHON) direct_launcher.py --output vit_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(BASE_PYTHON) direct_launcher.py --output resnet_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(BASE_PYTHON) rednet_launcher.py --output vit_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(BASE_PYTHON) rednet_launcher.py --output resnet_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(SAVE_PYTHON) save_launcher.py --output vit_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(SAVE_PYTHON) save_launcher.py --output resnet_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(BASE_PYTHON) scripts/conclude.py vit_direct.csv resnet_direct.csv vit_rednet.csv resnet_rednet.csv vit_save.csv resnet_save.csv

phy:
	$(BASE_PYTHON) direct_launcher.py --output vit_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin
	$(BASE_PYTHON) direct_launcher.py --output resnet_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin
	$(BASE_PYTHON) rednet_launcher.py --output vit_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin
	$(BASE_PYTHON) rednet_launcher.py --output resnet_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin
	$(SAVE_PYTHON) save_launcher.py --output vit_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin
	$(SAVE_PYTHON) save_launcher.py --output resnet_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin
	$(BASE_PYTHON) scripts/conclude.py vit_direct.csv resnet_direct.csv vit_rednet.csv resnet_rednet.csv vit_save.csv resnet_save.csv

virtFast:
	$(BASE_PYTHON) direct_launcher.py --output vit_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --fast True
	$(BASE_PYTHON) direct_launcher.py --output resnet_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --fast True
	$(BASE_PYTHON) rednet_launcher.py --output vit_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --fast True
	$(BASE_PYTHON) rednet_launcher.py --output resnet_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --fast True
	$(SAVE_PYTHON) save_launcher.py --output vit_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --fast True
	$(SAVE_PYTHON) save_launcher.py --output resnet_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --fast True
	$(BASE_PYTHON) scripts/conclude.py vit_direct.csv resnet_direct.csv vit_rednet.csv resnet_rednet.csv vit_save.csv resnet_save.csv

phyFast:
	$(BASE_PYTHON) direct_launcher.py --output vit_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin --fast True
	$(BASE_PYTHON) direct_launcher.py --output resnet_direct.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin --fast True
	$(BASE_PYTHON) rednet_launcher.py --output vit_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin --fast True
	$(BASE_PYTHON) rednet_launcher.py --output resnet_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin --fast True
	$(SAVE_PYTHON) save_launcher.py --output vit_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512 --phy record.bin --fast True
	$(SAVE_PYTHON) save_launcher.py --output resnet_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512 --phy record.bin --fast True
	$(BASE_PYTHON) scripts/conclude.py vit_direct.csv resnet_direct.csv vit_rednet.csv resnet_rednet.csv vit_save.csv resnet_save.csv

.PHONY: clean
clean:
	rm *.csv