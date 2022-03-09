using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;



namespace NeuralNetwork.Layers
{
    class InputStandardizingLayer : ILayer
    {
        public ILayer UnderlyingLayer { get;  }

        public double[] Mean { get;  }

        public double[] Stddev { get;  }

        public int LayerSize { get => UnderlyingLayer.LayerSize; }

        public int InputSize { get => UnderlyingLayer.InputSize; }

        public int BatchSize { get => UnderlyingLayer.BatchSize; set => UnderlyingLayer.BatchSize = value; }
        public Matrix<double> Activation { get => UnderlyingLayer.Activation;}

        public Matrix<double> WeightedError { get => UnderlyingLayer.WeightedError;}

        public InputStandardizingLayer(ILayer underlyingLayer, double[] mean, double[] stddev)
        {
            UnderlyingLayer = underlyingLayer;
            Mean = mean;
            Stddev = stddev;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            UnderlyingLayer.BackPropagate(upstreamWeightedErrors);
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }

        public void Propagate(Matrix<double> input)
        {
            input.MapIndexedInplace(
                (row,col, value) => {
                    double norm = (value - Mean[row]) / Stddev[row];
                    return norm; }
                );
            UnderlyingLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            // On update en modifiant la descente de gradient
            UnderlyingLayer.UpdateParameters();
        }
    }
}
