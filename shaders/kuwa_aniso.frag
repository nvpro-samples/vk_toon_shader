// by Jan Eric Kyprianidis <www.kyprianidis.com>
// https://github.com/MzHub/gpuakf

#version 450

// clang-format off
layout(set = 0, binding = 0) uniform sampler2D src;  
layout(set = 0, binding = 1) uniform sampler2D K0;  
layout(set = 0, binding = 2) uniform sampler2D tfm;
// clang-format on

layout(push_constant) uniform params_
{
  float radius;
  float q;
  float alpha;
};


layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;


const float PI = 3.14159265358979323846;
const int   N  = 8;

void main(void)
{
  vec2 src_size = vec2(textureSize(src, 0));
  vec2 uv       = gl_FragCoord.xy / src_size;

  vec4 m[8];
  vec3 s[8];
  for(int k = 0; k < N; ++k)
  {
    m[k] = vec4(0.0);
    s[k] = vec3(0.0);
  }

  float piN = 2.0 * PI / float(N);
  mat2  X   = mat2(cos(piN), sin(piN), -sin(piN), cos(piN));

  vec4  t = texture(tfm, uv);
  float a = radius * clamp((alpha + t.w) / alpha, 0.1, 2.0);
  float b = radius * clamp(alpha / (alpha + t.w), 0.1, 2.0);

  float cos_phi = cos(t.z);
  float sin_phi = sin(t.z);

  mat2 R  = mat2(cos_phi, -sin_phi, sin_phi, cos_phi);
  mat2 S  = mat2(0.5 / a, 0.0, 0.0, 0.5 / b);
  mat2 SR = S * R;

  int max_x = int(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
  int max_y = int(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

  for(int j = -max_y; j <= max_y; ++j)
  {
    for(int i = -max_x; i <= max_x; ++i)
    {
      vec2 v = SR * vec2(i, j);
      if(dot(v, v) <= 0.25)
      {
        vec4 c_fix = texture(src, uv + vec2(i, j) / src_size);
        vec3 c     = c_fix.rgb;
        for(int k = 0; k < N; ++k)
        {
          float w = texture(K0, vec2(0.5, 0.5) + v).x;

          m[k] += vec4(c * w, w);
          s[k] += c * c * w;

          v *= X;
        }
      }
    }
  }

  vec4 o = vec4(0.0);
  for(int k = 0; k < N; ++k)
  {
    m[k].rgb /= m[k].w;
    s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);

    float sigma2 = s[k].r + s[k].g + s[k].b;
    float w      = 1.0 / (1.0 + pow(255.0 * sigma2, 0.5 * q));

    o += vec4(m[k].rgb * w, w);
  }

  float alpha = texture(src, uv).a;
  fragColor   = vec4(o.rgb / o.w, alpha);
}
