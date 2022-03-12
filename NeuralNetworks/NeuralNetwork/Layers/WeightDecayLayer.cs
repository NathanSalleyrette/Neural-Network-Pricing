using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;


namespace NeuralNetwork.Layers
{
    class WeightDecayLayer : ILayer
    {
        public BasicStandardLayer StandardLayer { get; }

        public double DecayRate { get; }
        public int LayerSize { get => StandardLayer.LayerSize; }

        public int InputSize { get => StandardLayer.InputSize; }

        public int BatchSize { get => StandardLayer.BatchSize; set => StandardLayer.BatchSize = value; }

        public Matrix<double> Activation { get => StandardLayer.Activation; set => StandardLayer.Activation = value; }

        public Matrix<double> WeightedError { get => StandardLayer.WeightedError; set => StandardLayer.WeightedError = value; }

        public WeightDecayLayer(BasicStandardLayer basicStandardLayer, double decayRate)
        {
            StandardLayer = basicStandardLayer;
            DecayRate = decayRate;
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            StandardLayer.BackPropagate(upstreamWeightedErrors);
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
            StandardLayer.InitialWeights = StandardLayer.InitialWeights.Multiply(1.0 - DecayRate);
            StandardLayer.UpdateParameters();
        }
    }
}
