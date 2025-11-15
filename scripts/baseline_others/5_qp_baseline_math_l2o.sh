
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer CoordMathLSTM \
    --p-use --a-use \
    --p-norm "sigmoid" \
    --a-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 20 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --identical-dict \
    --global-training-steps 5000 \
    --save-dir training/nan_test/SGDlr0.001-T100-512400-QP-Math-L2O-SgleLoss-DetachState-sigmoid-ZeroX0