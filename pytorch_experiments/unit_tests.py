
import unittest
import torch
from model_architectures import (
    ConvolutionalProcessingBlock_Batch_Norm,
    ConvolutionalProcessingBlock_Batch_Norm_Residual_Connections,
    ConvolutionalDimensionalityReductionBlock_Batch_Norm,
    ConvolutionalDimensionalityReductionBlock_Batch_Norm_Residual_Connections
)

class TestBlocks(unittest.TestCase):
    def test_bn_convolutional_processing_block(self):
        input_shape = (1, 3, 64, 64)
        num_filters = 3
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1

        block = ConvolutionalProcessingBlock_Batch_Norm(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation
        )

        x = torch.randn(input_shape)
        output = block(x)
        expected_shape = (1, num_filters, 64, 64)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")
        #print(f"ConvolutionalProcessingBlock_Batch_Norm Output shape: {output.shape}")

    
    def test_res_bn_convolutional_processing_block(self):
        input_shape = (1, 3, 64, 64)
        num_filters = 3
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1

        block = ConvolutionalProcessingBlock_Batch_Norm_Residual_Connections(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation
        )

        x = torch.randn(input_shape)
        output = block(x)
        expected_shape = (1, num_filters, 64, 64)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")
        #print(f"ConvolutionalProcessingBlock_Batch_Norm_Residual_Connections Output Shape: {output.shape}")

    def test_bn_convolutional_dimensionality_reduction_block(self):
        input_shape = (1, 3, 64, 64)
        num_filters = 3
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1
        reduction_factor = 2 # 64 -> 32

        block = ConvolutionalDimensionalityReductionBlock_Batch_Norm(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
            reduction_factor=reduction_factor
        )

        x = torch.randn(input_shape)
        output = block(x)

        expected_shape = (1, num_filters, 32, 32)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")
        #print(f"ConvolutionalDimensionalityReductionBlock_Batch_Norm Output shape: {output.shape}")



    def test_res_bn_convolutional_dimensionality_reduction_block(self):
        input_shape = (1, 3, 64, 64)
        num_filters = 3
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1
        reduction_factor = 2 # 64 -> 32

        block = ConvolutionalDimensionalityReductionBlock_Batch_Norm_Residual_Connections(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
            reduction_factor=reduction_factor
        )

        x = torch.randn(input_shape)
        output = block(x)

        expected_shape = (1, num_filters, 32, 32)
        self.assertEqual(output.shape, expected_shape, "Output shape mismatch")
        #print(f"ConvolutionalDimensionalityReductionBlock_Batch_Norm Output shape: {output.shape}")


if __name__ == "__main__":
    unittest.main()
