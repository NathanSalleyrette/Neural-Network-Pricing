﻿using System;
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
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) => mat.Multiply(- learningRate);

        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) => mat.Multiply(-learningRate);

        public GradientAdjustmentType Type => GradientAdjustmentType.FixedLearningRate;
    }
}
