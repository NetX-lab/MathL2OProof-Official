# This script is used to train a model to test its inference peroformance on the quadratic problem with random orthogonality constraint.

# NOTE: Case 1 Scale 32-25
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.001 \
#     --e 100 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir inference/SGDlr0.001-T100-rand100orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 32 \
#     --output-dim 25 \
#     --lstm-hidden-size 1024 \
#     --test-length 5000 \
#     --load-mat \
#     --optimizee-dir ./optimizees/matdata/qp-rand-3225 \
#     --loss-save-path qp-rand \
#     --save-dir inference/SGDlr0.001-T100-rand100orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0

# NOTE: Case 2 Scale 512-400
# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --lstm-hidden-size 5120 \
#     --train-batch-size 10 \
#     --val-size 10 \
#     --init-lr 0.0000001 \
#     --e 500 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir inference/SGDlr0.0000001-T100-rand500orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --lstm-hidden-size 5120 \
#     --test-length 5000 \
#     --test-size 32 \
#     --test-batch-size 32 \
#     --load-mat \
#     --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
#     --loss-save-path qp-rand \
#     --save-dir inference/SGDlr0.0000001-T100-rand500orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


# Tune for Inference
# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --lstm-hidden-size 5120 \
#     --test-length 5000 \
#     --test-size 32 \
#     --test-batch-size 32 \
#     --load-mat \
#     --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
#     --loss-save-path qp-rand \
#     --save-dir training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr0.001--optimizer-training-steps100--unroll-length100

# python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
#     --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
#     --device "cuda:0" \
#     --input-dim 512 \
#     --output-dim 400 \
#     --lstm-hidden-size 5120 \
#     --test-length 5000 \
#     --test-size 32 \
#     --test-batch-size 32 \
#     --load-mat \
#     --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
#     --loss-save-path qp-rand \
#     --save-dir training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr0.0001--optimizer-training-steps100--unroll-length100


python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --loss-save-path qp-rand \
    --save-dir training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-05--optimizer-training-steps100--unroll-length100


python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --loss-save-path qp-rand \
    --save-dir training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-06--optimizer-training-steps100--unroll-length100

python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer CoordMathDNN --p-use --p-norm "sigmoid" \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --lstm-hidden-size 5120 \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --loss-save-path qp-rand \
    --save-dir training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-07--optimizer-training-steps100--unroll-length100