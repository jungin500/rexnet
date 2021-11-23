from loguru import logger
logger.info("Load libraries ...")

import torch
import onnx
from onnxsim import simplify
from torchinfo import summary
from argparse import ArgumentParser

from rexnetv1_lite import ReXNetV1_lite
from rexnetv1 import ReXNetV1

def main(args):
    weight_path = args.weight_path
    output_filename_base = args.output_base
    batch_size = args.batch_size  # Batch size must be 32 (not 16), don't know why
    is_dynamic_batch = args.dynamic_batch  # we need dynamic batch due to non-implicit input batch sizes
    # ONNX should accept dynamic batch so it can be used on inference time
    
    ONNX_OUTPUT_PATH = '%s_b%d%s.onnx' % (output_filename_base, batch_size, '_dynbatch' if is_dynamic_batch else '')
    
    logger.info("Creating model ...")
    model = ReXNetV1_lite(multiplier=1.0, classes=3)
    
    if weight_path is not None:
        logger.info("Loading weight ...")
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt)
    else:
        logger.warning("Weight file not loaded; exported ONNX will have no ability to classify items!")
        logger.warning("USE IT WITH CARE AND TESTING PURPOSES!")
        
        logger.warning("Appending _noweight into output model name")
        ONNX_OUTPUT_PATH = ONNX_OUTPUT_PATH.replace('.onnx', '_noweight.onnx')
    
    model = torch.nn.Sequential(
        model,
        torch.nn.Sigmoid()
    )
    
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 128)  # Input batch size is IMPORTANT!
    
    if weight_path is not None:
        summary(model, list(dummy_input.shape), device='cpu', depth=4)
    
    logger.info("Exporting to ONNX ...")
    torch.onnx._export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        } if is_dynamic_batch else None,
        # dynamic_axes=None,
        opset_version=11
    )
    logger.info("Done writing onnx file, simplifying ...")
    
    # Simplify
    input_shapes = {
        'images': list(dummy_input.shape)
    } if is_dynamic_batch else None
    
    logger.info("Loading ONNX ...")
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    
    logger.info("Simplifying ONNX ...")
    model_simp, check = simplify(onnx_model,
                                 dynamic_input_shape=is_dynamic_batch,
                                 input_shapes=input_shapes)
    assert check, "SimpONNX can't be verified"
    
    logger.info("Saving ONNX ...")
    onnx.save(model_simp, ONNX_OUTPUT_PATH)
    logger.info("Successfully simplified and saved onnx model!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-base', '-o', type=str, default='rexnet', help='Output filename base (e.g. \'rexnet\' -> \'rexnet_b4_dynbatch.onnx\')')
    parser.add_argument('--weight-path', '-w', type=str, default=None, help='ReXNet weight path')
    parser.add_argument('--batch-size', '-b', type=int, required=True, help='Batch size to create ONNX model.')
    parser.add_argument('--dynamic-batch', '-i', action='store_true', help='Enable dynamic batch ONNX model, with parameter \'batch_size\'.')
    args = parser.parse_args()
    main(args)