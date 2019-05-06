Before running code, make a new directory named "save"

To run SimBA (pixel attack):
python run_simba.py --num_iters 30000 --pixel_attack

To run SimBA-DCT (low frequency attack):
python run_simba.py --num_iters 30000 --freq_dims 28 --order strided --stride 7

For targeted attack, add flag --targeted and change --num_iters to 100000
