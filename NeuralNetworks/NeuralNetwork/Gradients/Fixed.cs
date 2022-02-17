using System;
using System.Collections.Generic;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System.Text;
using MathNet.Numerics.LinearAlgebra;


namespace NeuralNetwork.Gradients
{
    public class Fixed : IGradient
    {
        private double learningRate;
        public Fixed(FixedLearningRateParameters gradient)
        {
            learningRate = gradient.LearningRate;
        }
        public Func<Matrix<double>, Matrix<double>> Apply => (mat) => learningRate * mat;
        public GradientAdjustmentType Type => GradientAdjustmentType.FixedLearningRate;
    }
}
