# python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
#     --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
#     --device "cuda:1" \
#     --input-dim 64 \
#     --output-dim 50 \
#     --lstm-hidden-size 2048 \
#     --train-batch-size 32 \
#     --val-size 32 \
#     --init-lr 0.0000001 \
#     --e 100 \
#     --optimizer-training-steps 100    --unroll-length 100 \
#     --global-training-steps 5000 \
#     --save-dir ablation_fullsparse_sumloss/SGDlr0.0000001-T100-rand100orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0000001 \
    --e 50 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss/SGDlr0.0000001-T100-rand50orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0000001 \
    --e 25 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss/SGDlr0.0000001-T100-rand25orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0000001 \
    --e 5 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss/SGDlr0.0000001-T100-rand5orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



python main_train_anlys.py --config ./configs/1_qp_training_train_anlys.yaml \
    --p-use --optimizer CoordMathDNN --p-norm "sigmoid" \
    --device "cuda:1" \
    --input-dim 64 \
    --output-dim 50 \
    --lstm-hidden-size 2048 \
    --train-batch-size 32 \
    --val-size 32 \
    --init-lr 0.0000001 \
    --e 1 \
    --optimizer-training-steps 100    --unroll-length 100 \
    --global-training-steps 5000 \
    --save-dir ablation_fullsparse_sumloss/SGDlr0.0000001-T100-rand1orth-6450-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0


# sh scripts/2_qp_l2o_p_ablation_e_128100.sh