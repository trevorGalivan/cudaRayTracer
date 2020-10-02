#version 460 core
out vec4 FragColor;
  
in vec2 TexCoord;

layout(binding=1) uniform sampler2D textureVar;
void main()
{
    FragColor = texture(textureVar, TexCoord);
}