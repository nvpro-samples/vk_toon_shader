// by Jan Eric Kyprianidis <www.kyprianidis.com>
#version 450

layout(set = 0, binding = 0) uniform sampler2D src;
layout(location = 0) out vec4 fragColor;
layout(location = 0) in vec2 fragCoord;


void main(void)
{
  vec2 src_size = vec2(textureSize(src, 0));
  vec2 uv       = gl_FragCoord.xy / src_size;
  vec2 d        = 1.0 / src_size;

  vec3 c = texture(src, uv).xyz;
  vec3 u = (-1.0 * texture(src, uv + vec2(-d.x, -d.y)).xyz + -2.0 * texture(src, uv + vec2(-d.x, 0.0)).xyz
            + -1.0 * texture(src, uv + vec2(-d.x, d.y)).xyz + +1.0 * texture(src, uv + vec2(d.x, -d.y)).xyz
            + +2.0 * texture(src, uv + vec2(d.x, 0.0)).xyz + +1.0 * texture(src, uv + vec2(d.x, d.y)).xyz)
           / 4.0;

  vec3 v = (-1.0 * texture(src, uv + vec2(-d.x, -d.y)).xyz + -2.0 * texture(src, uv + vec2(0.0, -d.y)).xyz
            + -1.0 * texture(src, uv + vec2(d.x, -d.y)).xyz + +1.0 * texture(src, uv + vec2(-d.x, d.y)).xyz
            + +2.0 * texture(src, uv + vec2(0.0, d.y)).xyz + +1.0 * texture(src, uv + vec2(d.x, d.y)).xyz)
           / 4.0;

  fragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);
}
