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
    }
}