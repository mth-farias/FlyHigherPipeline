using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class NormalizeSleapOutput
{
    private Size cropSize;
    public Size CropSize
    {
        get { return cropSize; }
        set { cropSize = value; }
    }

    [Description("Scaling factor that will be applied after normalizing the positions to the size of the crop.")]
    private float scalingFactor = 1;
    public float ScalingFactor
    {
        get { return scalingFactor; }
        set { scalingFactor = value; }
    }


    public IObservable<NormalizedPose> Process(IObservable<Pose> source)
    {
        return source.Select(value => {
            return new NormalizedPose(value, cropSize, scalingFactor);
        });
    }



    public class NormalizedPose : Pose
    {
        public NormalizedPose(Pose pose, Size cropSize, float scalingFactor) : base(pose.Image)
        {
            var normPose = new Pose(this.Image);
            normPose.Centroid = this.Centroid;
            foreach (var bodypart in pose){
                var normalizedBodyPart = new BodyPart();
                normalizedBodyPart.Name = bodypart.Name;
                normalizedBodyPart.Confidence = bodypart.Confidence;
                normalizedBodyPart.Position  = new Point2f(
                    bodypart.Position.X / cropSize.Width * scalingFactor,
                    bodypart.Position.Y / cropSize.Height * scalingFactor
                    );
                normPose.Add(normalizedBodyPart);
            }
            Normalized = normPose;
        }

        public Pose Normalized { get; set; }

    }
}
