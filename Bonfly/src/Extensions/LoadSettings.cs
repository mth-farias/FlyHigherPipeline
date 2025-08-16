using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using YamlDotNet.Serialization;
using System.IO;
using YamlDotNet.Serialization.NamingConventions;
using YamlDotNet.Core;
using AutomaticGenerators;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadSettings
{
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string Path { get; set;}

    public IObservable<BonFlyStimuli> Process()
    {
        return Observable.Defer(() =>
        {
            BonFlyStimuli settings;
            using (var reader = new StreamReader(Path))
            {
                var parser = new MergingParser(new Parser(reader));

                var deserializer = new DeserializerBuilder()
                    .WithNamingConvention(CamelCaseNamingConvention.Instance)
                    .Build();
                settings =  deserializer.Deserialize<BonFlyStimuli>(parser);
            }
            return Observable.Return(settings);
        });
    }
}
