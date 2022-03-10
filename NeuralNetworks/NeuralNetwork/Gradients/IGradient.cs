using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;


namespace NeuralNetwork.Gradients
{
    public interface IGradient
    {
        Func<Matrix<double>, Matrix<double>> VWeight { get;  }
        Func<Matrix<double>, Matrix<double>> VBias { get; }
        GradientAdjustmentType Type { get;  }

    }
}
