using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;

namespace NeuralNetwork.Gradients
{
    internal static class GradientFactory
    {
        public static IGradient Build(IGradientAdjustmentParameters gradientAdjustment)
        {
            switch (gradientAdjustment.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    return new Fixed(gradientAdjustment as FixedLearningRateParameters);

                //case GradientAdjustmentType.Momentum:
                //    return new Identity();

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