MODEL_PATH=$1
echo "$MODEL_PATH"
bash verl/recipe/l2s/eval_aime24.sh "$MODEL_PATH"
bash verl/recipe/l2s/eval_amc.sh "$MODEL_PATH"
bash verl/recipe/l2s/eval_olympiad_bench.sh "$MODEL_PATH"
bash verl/recipe/l2s/eval_math.sh "$MODEL_PATH"
bash verl/recipe/l2s/eval_minerva.sh "$MODEL_PATH"