using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;
using System;
using NeuralNetwork.Layers;

namespace NeuralNetwork.Serialization
{
    public static class NetworkSerializer
    {
        public static SerializedNetwork Serialize(INetwork network)
        {
            var sNetwork = new SerializedNetwork();
            sNetwork.BatchSize = network.BatchSize;
            var Layers = network.Layers;
            var SerializedLayers = new ISerializedLayer[Layers.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                var layer = Layers[i];
                switch (layer)
                {
                    case BasicStandardLayer bl:
                        
                        var Serializedlayer = new SerializedStandardLayer(bl.InitialBias.ToColumnMajorArray(), bl.InitialWeights.ToArray(), bl.Activator.Type, bl.GradientAdjustment);
                        SerializedLayers[i] = Serializedlayer;
                        break;

                    default:
                        throw new InvalidOperationException();
                }
            }

            sNetwork.SerializedLayers = SerializedLayers;
            return sNetwork;
        }
    }
}
