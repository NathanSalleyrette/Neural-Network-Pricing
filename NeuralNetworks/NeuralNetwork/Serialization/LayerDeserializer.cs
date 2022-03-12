using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Serialization
{
    internal static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var standardSerialized = serializedLayer as SerializedStandardLayer;
                    return DeserializeBasicStandardLayer(standardSerialized, batchSize);

                case LayerType.L2Penalty:
                    var l2PenaltySerialized = serializedLayer as SerializedL2PenaltyLayer;
                    return DeserializeL2PenaltyLayer(l2PenaltySerialized, batchSize);

                case LayerType.InputStandardizing:
                    var inputStandardSerialized = serializedLayer as SerializedInputStandardizingLayer;
                    return DeserializeInputStandardizingLayer(inputStandardSerialized, batchSize);

                case LayerType.Dropout:
                    var dropoutSerialized = serializedLayer as SerializedDropoutLayer;
                    return DeserializeDropoutLayer(dropoutSerialized, batchSize);

                case LayerType.WeightDecay:
                    var weightDecaySerialized = serializedLayer as SerializedWeightDecayLayer;
                    return DeserializeWeightDecayLayer(weightDecaySerialized, batchSize);

                case LayerType.L1Penalty:
                    var l1PenaltySerialized = serializedLayer as SerializedL1PenaltyLayer;
                    return DeserializeL1PenaltyLayer(l1PenaltySerialized, batchSize);

                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");
            }
        }


        private static ILayer DeserializeBasicStandardLayer(SerializedStandardLayer standardSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfArray(standardSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { standardSerialized.Bias });
            var inputSize = weights.RowCount;
            var layerSize = weights.ColumnCount;
            var gradientAdjustment = GradientFactory.Build(standardSerialized.GradientAdjustmentParameters, weights, bias, batchSize);
            var activator = ActivatorFactory.Build(standardSerialized.ActivatorType);
            return new BasicStandardLayer(weights, gradientAdjustment, bias, batchSize, activator, standardSerialized.GradientAdjustmentParameters);
        }

        private static ILayer DeserializeL2PenaltyLayer(SerializedL2PenaltyLayer l2PenaltySerialized, int batchSize)
        {
            // On déséréalise tout ici ? ou dans L2Layer ? 
            var penalty = l2PenaltySerialized.PenaltyCoefficient;
            switch (l2PenaltySerialized.UnderlyingSerializedLayer.Type)
            {
                case LayerType.Standard:
                    var basicStandardLayer = DeserializeBasicStandardLayer(l2PenaltySerialized.UnderlyingSerializedLayer as SerializedStandardLayer, batchSize);
                    return new L2Layer(basicStandardLayer as BasicStandardLayer, penalty);

                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");

            }

        }

        private static ILayer DeserializeL1PenaltyLayer(SerializedL1PenaltyLayer l1PenaltySerialized, int batchSize)
        {
            // On déséréalise tout ici ? ou dans L2Layer ? 
            var penalty = l1PenaltySerialized.PenaltyCoefficient;
            switch (l1PenaltySerialized.UnderlyingSerializedLayer.Type)
            {
                case LayerType.Standard:
                    var basicStandardLayer = DeserializeBasicStandardLayer(l1PenaltySerialized.UnderlyingSerializedLayer as SerializedStandardLayer, batchSize);
                    return new L1Layer(basicStandardLayer as BasicStandardLayer, penalty);

                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");

            }

        }

        private static ILayer DeserializeWeightDecayLayer(SerializedWeightDecayLayer weightDecaySerialized, int batchSize)
        {
            var decayRate = weightDecaySerialized.DecayRate;
            switch (weightDecaySerialized.UnderlyingSerializedLayer.Type)
            {
                case LayerType.Standard:
                    var basicStandardLayer = DeserializeBasicStandardLayer(weightDecaySerialized.UnderlyingSerializedLayer as SerializedStandardLayer, batchSize);
                    return new L2Layer(basicStandardLayer as BasicStandardLayer, decayRate);

                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");

            }
        }

        private static ILayer DeserializeInputStandardizingLayer(SerializedInputStandardizingLayer inputStandardizingLayer, int batchSize)
        {
            var mean = inputStandardizingLayer.Mean;
            var stddev = inputStandardizingLayer.StdDev;
            var underlyingLayer = Deserialize(inputStandardizingLayer.UnderlyingSerializedLayer, batchSize);

            return new InputStandardizingLayer(underlyingLayer, mean, stddev);

        }

        private static ILayer DeserializeDropoutLayer(SerializedDropoutLayer dropoutSerialized,int batchSize)
        {
            var keepProbability = dropoutSerialized.KeepProbability;
            var layerSize = dropoutSerialized.LayerSize;

            return new DropoutLayer(layerSize, keepProbability,  batchSize);
        }

    }
}