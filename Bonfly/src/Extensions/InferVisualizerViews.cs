using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("Infers the current view of the subject using the highest confidence of the available bodyparts.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class InferVisualizerViews
{

    private string bodyPart_Left = "L";
    public string BodyPart_Left
    {
        get { return bodyPart_Left; }
        set { bodyPart_Left = value; }
    }

    private string bodyPart_Right = "R";
    public string BodyPart_Right
    {
        get { return bodyPart_Right; }
        set { bodyPart_Right = value; }
    }

    private string bodyPart_Top = "T";
    public string BodyPart_Top
    {
        get { return bodyPart_Top; }
        set { bodyPart_Top = value; }
    }

    private string bodyPart_Bottom = "B";
    public string BodyPart_Bottom
    {
        get { return bodyPart_Bottom; }
        set { bodyPart_Bottom = value; }
    }

    private string bodyPart_Head = "H";
    public string BodyPart_Head
    {
        get { return bodyPart_Head; }
        set { bodyPart_Head = value; }
    }

    private string bodyPart_Thorax = "Trx";
    public string BodyPart_Thorax
    {
        get { return bodyPart_Thorax; }
        set { bodyPart_Thorax = value; }
    }

    private string bodyPart_Abdomen = "Abd";
    public string BodyPart_Abdomen
    {
        get { return bodyPart_Abdomen; }
        set { bodyPart_Abdomen = value; }
    }

    private string bodyPart_LeftWing = "Lw";
    public string BodyPart_LeftWing
    {
        get { return bodyPart_LeftWing; }
        set { bodyPart_LeftWing = value; }
    }

    private string bodyPart_RightWing = "Rw";
    public string BodyPart_RightWing
    {
        get { return bodyPart_RightWing; }
        set { bodyPart_RightWing = value; }
    }


    public IObservable<VisualizerViews> Process(IObservable<Pose> source)
    {
        return source.Select(value => {
            var bestView = InferBestView(value);
            var cleanView = InferCleanView(value, bestView);
            var horizonView = InferHorizon(value, cleanView);
            return new VisualizerViews() { BestView = bestView, CleanView = cleanView, HorizonView = horizonView};
        });
    }

    static readonly string[] BestViews = { "Left", "Right", "Top", "Bottom" };

    public InferredView InferBestView(Pose value)
    {
        BodyPart[] ViewBodyParts = {
            value[bodyPart_Left],
            value[bodyPart_Right],
            value[bodyPart_Top],
            value[bodyPart_Bottom]};
        var Confidence = ViewBodyParts.Select(x => x.Confidence).ToArray();
        float MaximumConfidence = Confidence.Max();
        int ArgMax = Array.IndexOf(Confidence, MaximumConfidence);
        var BestView = ArgMax == -1 ? string.Empty : BestViews[ArgMax];
        var BestCoordinate = ArgMax == -1 ? new Point2f(float.NaN, float.NaN) : ViewBodyParts[ArgMax].Position;
        return new InferredView { ViewName = BestView, Position = BestCoordinate, Confidence = MaximumConfidence };
    }

    public InferredView InferCleanView(Pose value, InferredView bestView)
    {

            BodyPart[] ViewBodyParts = {
            value[bodyPart_Head],
            value[bodyPart_Thorax],
            value[bodyPart_Abdomen]};
            var N_visible = ViewBodyParts.Count(bp => !float.IsNaN(bp.Position.X));

            float conf = float.NaN;
            string view = string.Empty;

            if (!float.IsNaN(bestView.Position.X)){
                if (N_visible == 3){
                    view = bestView.ViewName;
                    conf = bestView.Confidence;
                }
                else{
                    view = "Angle";
                    conf = 1;
                }
            }
            if (N_visible == 1){
                view = "Vertical";
                conf = 1;
            }
            else if (N_visible == 2){
                view = "Angle";
                conf = 1;
            }
            return new InferredView { ViewName = view, Position = bestView.Position, Confidence = conf };
    }

    public Tuple<Point2f, Point2f> InferHorizon(Pose value, InferredView cleanView){

        var view = cleanView.ViewName;

        var forwPoint = new Point2f();
        var backPoint = new Point2f();

        if (view == "Vertical"){
            forwPoint = value[bodyPart_Head].Position;
            backPoint = value[bodyPart_Head].Position;
        }
        else if (view == "Angle"){
            forwPoint = value[bodyPart_Head].Position;
            backPoint = value[bodyPart_Abdomen].Position;
        }
        else{
            forwPoint = cleanView.Position;
            backPoint = value[bodyPart_Thorax].Position;
        }

        return Tuple.Create(forwPoint, backPoint);
    }

    public struct VisualizerViews{
        public InferredView BestView;
        public InferredView CleanView;
        public Tuple<Point2f, Point2f> HorizonView;
    }

    public struct InferredView{
        public string ViewName;
        public Point2f Position;
        public float Confidence;
    }
}
