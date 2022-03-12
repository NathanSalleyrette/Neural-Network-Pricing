using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;


namespace NeuralNetwork.Layers
{
    class L2Layer : ILayer
    {
        public BasicStandardLayer StandardLayer { get; }

        public double PenaltyCoefficient { get; }
        public int LayerSize { get => StandardLayer.LayerSize; }

        public int InputSize { get => StandardLayer.InputSize; }

        public int BatchSize { get => StandardLayer.BatchSize; set => StandardLayer.BatchSize = value; }

        public Matrix<double> Activation { get => StandardLayer.Activation; set => StandardLayer.Activation = value; }

        public Matrix<double> WeightedError { get => StandardLayer.WeightedError; set => StandardLayer.WeightedError = value; }

        public L2Layer(BasicStandardLayer basicStandardLayer, double penalty)
        {
            StandardLayer = basicStandardLayer;
            PenaltyCoefficient = penalty;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            StandardLayer.BackPropagate(upstreamWeightedErrors);
            StandardLayer.GradientWeight = StandardLayer.GradientWeight.Add(StandardLayer.InitialWeights.Multiply(PenaltyCoefficient));
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }

        public void Propagate(Matrix<double> input)
        {
            StandardLayer.Propagate(input);
        }

        public void UpdateParameters()
        {
            // On update en modifiant la descente de gradient
            StandardLayer.UpdateParameters();
        }
    }
}
