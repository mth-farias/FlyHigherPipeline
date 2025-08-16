#version 400
in vec2 texCoord;
out vec4 fragColor;

uniform vec4 colorOn = vec4(1);
uniform vec4 colorOff = vec4(1);
uniform float contrast = 0.5;
uniform vec2 randomSeed = vec2(12.9898,78.233);

float rand(vec2 coord){
  return fract(sin(dot(coord.xy , randomSeed)) * 43758.5453);
}

void main()
{
  if (length(2 * texCoord - 1) > 1)
    discard;
  fragColor = rand(texCoord) < contrast  ? colorOn : colorOff;
}