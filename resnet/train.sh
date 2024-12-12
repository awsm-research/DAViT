python main.py \
    --model_name_or_path=wide_resnet101 \
    --model_name=wide_resnet101.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../data/chest_xray/train \
    --eval_data_file=../data/chest_xray/val \
    --test_data_file=../data/chest_xray/test \
    --epochs 100 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee wide_resnet101.log

python main.py \
    --model_name_or_path=resnext101 \
    --model_name=resnext101.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../data/chest_xray/train \
    --eval_data_file=../data/chest_xray/val \
    --test_data_file=../data/chest_xray/test \
    --epochs 100 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee resnext101.log

python main.py \
    --model_name_or_path=resnet152 \
    --model_name=resnet152.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../data/chest_xray/train \
    --eval_data_file=../data/chest_xray/val \
    --test_data_file=../data/chest_xray/test \
    --epochs 100 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee resnet152.log

python main.py \
    --model_name_or_path=resnet50 \
    --model_name=resnet50.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../data/chest_xray/train \
    --eval_data_file=../data/chest_xray/val \
    --test_data_file=../data/chest_xray/test \
    --epochs 100 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee resnet50.log