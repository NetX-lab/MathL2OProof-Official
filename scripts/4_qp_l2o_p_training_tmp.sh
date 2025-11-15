
python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --train-batch-size 10 \
    --val-size 10 \
    --init-lr 0.001 \
    --e 5000 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --identical-dict \
    --fixed-dict \
    --global-training-steps 5000 \
    --save-dir training/SingleM/SGDlr0.001-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM