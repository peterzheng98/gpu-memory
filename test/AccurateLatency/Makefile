all:
	$(BASE_PYTHON) drdna_launcher.py --output vit_drdna.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(BASE_PYTHON) drdna_launcher.py --output resnet_drdna.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(BASE_PYTHON) rednet_launcher.py --output vit_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(BASE_PYTHON) rednet_launcher.py --output resnet_rednet.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(SAVE_PYTHON) save_launcher.py --output vit_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model vit --batch_size 512
	$(SAVE_PYTHON) save_launcher.py --output resnet_save.csv --root_dir=/home/zwx/gpu-ecc-motivation/datas --model resnet --batch_size 512
	$(BASE_PYTHON) scripts/conclude.py vit_drdna.csv vit_rednet.csv vit_save.csv
	$(BASE_PYTHON) scripts/conclude.py resnet_drdna.csv resnet_rednet.csv resnet_save.csv

clean:
	rm *.csv