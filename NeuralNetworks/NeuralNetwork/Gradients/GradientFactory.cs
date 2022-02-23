using NeuralNetwork.Common.GradientAdjustmentParameters;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork.Gradients
{
    internal static class GradientFactory
    {
        public static IGradient Build(IGradientAdjustmentParameters gradientAdjustment, Matrix<double> weight, Matrix<double> bias, int batchSize)
        {
            switch (gradientAdjustment.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    return new Fixed(gradientAdjustment as FixedLearningRateParameters);

                case GradientAdjustmentType.Momentum:

                    // Remove Batch Size, it's not used by Momentum
                    return new Momentum(gradientAdjustment as MomentumParameters, weight, bias, batchSize);

                //case GradientAdjustmentType.Nesterov:
                //    return;

                //case GradientAdjustmentType.Adam:
                //    return;


                default:
                    throw new InvalidOperationException("Unknown activator type: " + gradientAdjustment.ToString());
            }
        }
    }
}