cuda=$1
args=$2

save_dir=./results/demo
model=v5

mkdir ./results
mkdir $save_dir

# For optimizer-based method.
CUDA_VISIBLE_DEVICES=$cuda nohup python train_optim.py \
-cfg=demo.yaml \
-s=$save_dir \
-n=v5-combine-demo \
$args \
>$save_dir/test.log 2>&1 &

echo "Training... Patch & logs will be saved to $save_dir/"

# use 'python train_optim.py -h' for detailed supports of the arguments.


########## You can run train_fgsm.py script from fgsm-based methods.
#CUDA_VISIBLE_DEVICES=$cuda nohup python train_fgsm.py \
#-cfg=method/v5-bim-combine.yaml \
#-s=$save_dir \
#-n=demo \
#>$save_dir/test.log 2>&1 &

# use 'python train_fgsm.py -h' for detailed supports of the arguments.