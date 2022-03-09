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

                    case L2Layer l2l:
                        var bsl = l2l.StandardLayer;
                        var SerializedStandardlayer = new SerializedStandardLayer(bsl.InitialBias.ToColumnMajorArray(), bsl.InitialWeights.ToArray(), bsl.Activator.Type, bsl.GradientAdjustment);

                        var SerializedL2layer = new SerializedL2PenaltyLayer(SerializedStandardlayer, l2l.PenaltyCoefficient);
                        SerializedLayers[i] = SerializedL2layer;
                        
                        break;

                    case InputStandardizingLayer isl:

                        var usl = isl.UnderlyingLayer as BasicStandardLayer;
                        var SerializedUsl = new SerializedStandardLayer(usl.InitialBias.ToColumnMajorArray(), usl.InitialWeights.ToArray(), usl.Activator.Type, usl.GradientAdjustment);

                        var SerializedISLLayer = new SerializedInputStandardizingLayer(SerializedUsl, isl.Mean, isl.Stddev);
                        SerializedLayers[i] = SerializedISLLayer;

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
