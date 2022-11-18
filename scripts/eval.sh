cuda=$1
save=$2

save_dir=$PROJECT_DIR/eval/inria/demo
PROJECT_DIR=$(pwd)
targets=(eval/coco80 eval/coco91)

for config in ${targets[@]}
do
  echo "Config file: $config"
  echo "Evaluating the adversarial patch: $patch_path"

  cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py \
  -p $patch_path \
  -cfg ./configs/$config \
  -lp $PROJECT_DIR/data/INRIAPerson/Test/labels \
  -dr $PROJECT_DIR/data/INRIAPerson/Test/pos \
  -s $save_dir \
  -e 0 &"

  echo $cmd
  eval $cmd
  sleep 2
done
# Use an absolute path for -lp(label path), -dr(preprocesser directory) and -s(save path).
# For detailed supports of the arguments:
#python evaluate.py -h
