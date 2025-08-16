using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("Generates a virtual loom radius based on the linear approaching velocity of the virtual looming object. See Ache et al. 2019 for details")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CalculateBaselineRadius
{
    [Description("The maximum radius of the loom (in centimeters).")]
    private float maxRadius = 12.5f;
    public float MaxRadius
    {
        get { return maxRadius; }
        set { maxRadius = value; }
    }

    [Description("The distance of the subject to the monitor (in centimeters).")]
    private float monitorDistance = 18;
    public float MonitorDistance
    {
        get { return monitorDistance; }
        set { monitorDistance = value; }
    }


    [Description("The initial radius of the loom (in centimeters).")]
    private float initialRadius = 0.5f;
    public float InitialRadius
    {
        get { return initialRadius; }
        set { initialRadius = value; }
    }


    [Browsable(false)]
    [Description("The initial distance of the virtual loom to the subject (in centimeters).")]
    private float initialVirtualDistance = 12.5f;
    public float InitialVirtualDistance
    {
        get { return initialVirtualDistance; }
        set { initialVirtualDistance = value; }
    }

    [Description("The linear approaching velocity of the virtual looming object (in cm/s).")]
    private float linearApproachingVelocity = 25;
    public float LinearApproachingVelocity
    {
        get { return linearApproachingVelocity; }
        set { linearApproachingVelocity = value; }
    }


    public IObservable<float> Process(IObservable<float> source)
    {
        return source.Select(value => {
            var t = value > -Single.Epsilon ? -Single.Epsilon : value;
            var theta = 2 * Math.Atan(InitialRadius / (-LinearApproachingVelocity * t));
            var radius = Math.Tan(theta/2) * MonitorDistance;
            radius = radius > MaxRadius ? MaxRadius : radius;
            return (float) radius;
        });
    }
}