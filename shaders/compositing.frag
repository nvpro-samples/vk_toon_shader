#version 450
//layout(set = 0, binding = 0) uniform sampler2D inTxt;

layout(set = 0, binding = 0) uniform sampler2D iChannel0;  // Ray tracer out
layout(set = 0, binding = 1) uniform sampler2D iChannel1;  // Normal & depth Contour
layout(set = 0, binding = 2) uniform sampler2D iChannel2;  // Object Contour

layout(push_constant) uniform params_
{
  vec3 backgroundColor;
  int  setBackground;
  vec3 lineColor;
};

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;

void main()
{
  vec4 color = texture(iChannel0, fragCoord.st);

  // White backgound
  if(setBackground > 0)
    color.xyz = mix(color.xyz, backgroundColor, 1.0 - texture(iChannel0, fragCoord.st).a);
  //
  vec4 ct1 = texture(iChannel1, fragCoord.st);
  vec4 ct2 = texture(iChannel2, fragCoord.st);

  color.xyz = mix(color.xyz, lineColor, ct1.r);
  color.xyz = mix(color.xyz, lineColor, ct2.r);  // outline contour over inline

  fragColor = color;
}
