# This script is used to run the inference for GD, LISTA, and our proposed L2O method for QP problem.

# Inference Case 1: Fixed M

# GD
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400-fixedM \
    --optimizer ProximalGradientDescent \
    --input-dim 512 \
    --output-dim 400 \
    --test-length 5000 \
    --load-mat \
    --identical-dict \
    --fixed-dict \
    --loss-save-path qp-rand-FixedM \
    --save-dir training/FixedM/GD512400-FixedM \
    --device "cuda:0"


# Our L2O
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
    --identical-dict \
    --fixed-dict \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400-fixedM \
    --loss-save-path qp-rand-FixedM \
    --save-dir training/FixedM/SGDlr0.001-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM



# LISTA-CPSS
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer LISTACPSS \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --rho 0.2 \
    --lamb 0.4 \
    --identical-dict \
    --fixed-dict \
    --w-shared \
    --theta-shared \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400-fixedM \
    --loss-save-path qp-rand-FixedM \
    --save-dir training/FixedM/LISTA-CPSS-Shared-FixM-lr0.001


python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer LISTACPSS \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --rho 0.2 \
    --lamb 0.4 \
    --identical-dict \
    --fixed-dict \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400-fixedM \
    --loss-save-path qp-rand-FixedM \
    --save-dir training/FixedM/LISTA-CPSS-NotShared-FixM-lr0.001


# Inference Case 2: Random M
# NOTE 1: For LISTA, we initialize it with mean M of all samples.
# NOTE 2: The fixed-dict is used for generateting a fixed random M before training, for LISTA initialization.
# GD
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --optimizer ProximalGradientDescent \
    --input-dim 512 \
    --output-dim 400 \
    --test-length 5000 \
    --load-mat \
    --loss-save-path qp-rand \
    --save-dir training/GD512400 \
    --device "cuda:0"


# Our L2O
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
    --save-dir training/SGDlr0.001-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0



# LISTA-CPSS
python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer LISTACPSS \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --rho 0.2 \
    --lamb 0.4 \
    --w-shared \
    --theta-shared \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --loss-save-path qp-rand \
    --save-dir training/LISTA-CPSS-Shared-lr0.001


python main_train_anlys.py --config ./configs/2_qp_testing.yaml \
    --optimizer LISTACPSS \
    --device "cuda:0" \
    --input-dim 512 \
    --output-dim 400 \
    --rho 0.2 \
    --lamb 0.4 \
    --test-length 5000 \
    --test-size 32 \
    --test-batch-size 32 \
    --load-mat \
    --optimizee-dir ./optimizees/matdata/qp-rand-512400 \
    --loss-save-path qp-rand \
    --save-dir training/LISTA-CPSS-NotShared-lr0.001