
# NOTE: GD 10000 iterations
python main_train_nn.py --config ./configs/1_qp_training_train_anlys.yaml \
    --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
    --device "cuda:0" \
    --lstm-hidden-size 20 \
    --global-training-steps 1 \
    --train-batch-size 200 \
    --test-batch-size 200 \
    --init-lr 0.000001 \
    --e 100 \
    --optimizer-training-steps 20000    --unroll-length 20000 \
    --save-dir inference/cnn/timecost/SGD-logtime20000  \
    --meta-optimizer SGD \
    --step-size 0.01


# NOTE: L2O 10000 iterations
# python main_train_nn.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --lstm-hidden-size 20 \
#     --global-training-steps 200 \
#     --train-batch-size 200 \
#     --test-batch-size 200 \
#     --init-lr 0.000001 \
#     --e 100 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --save-dir inference/cnn/timecost/SGDlr0.000001-T100-rand100orth-MINST200SGDStepS01-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-logtime200 \
#     --meta-optimizer SGD \
#     --step-size 0.01

# python main_train_nn.py --config ./configs/2_qp_testing.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --lstm-hidden-size 20 \
#     --test-length 5000 \
#     --train-batch-size 6000 \
#     --test-batch-size 600 \
#     --loss-save-path cnn-mnist \
#     --save-dir inference/cnn/SGDlr0.0000001-T100-rand100orth-MINST-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0