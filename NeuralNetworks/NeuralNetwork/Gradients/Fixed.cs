using System;
using System.Collections.Generic;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System.Text;
using MathNet.Numerics.LinearAlgebra;


namespace NeuralNetwork.Gradients
{
    public class Fixed : IGradient
    {
        public double LearningRate { get; }
        public Fixed(FixedLearningRateParameters gradient)
        {
            LearningRate = gradient.LearningRate;
        }
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) => mat.Multiply(- LearningRate);

        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) => mat.Multiply(- LearningRate);

        public GradientAdjustmentType Type => GradientAdjustmentType.FixedLearningRate;
    }
}
