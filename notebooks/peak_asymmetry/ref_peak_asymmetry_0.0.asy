
import settings;
import solids;
size(300);
outformat="png";
defaultshininess = 0.8;
currentlight = light(0,0,400);

// Camera information
currentprojection=orthographic (
camera=(8,5,4),
up=(0,0,1),
target=(2.1125, 2.1125, 2.1125000000000003),
zoom=0.5
);

// Basic function for drawing spheres
void drawSpheres(triple[] C, real R, pen p=currentpen){
  for(int i=0;i<C.length;++i){
    draw(sphere(C[i],R).surface(
                        new pen(int i, real j){return p;}
                        )
    );
  }
}

// Draw a sphere without light
void drawSpheres_nolight(triple[] C, real R, pen p=currentpen){
  material nlpen = material(diffusepen=opacity(1.0), emissivepen=p, shininess=0);
  for(int i=0;i<C.length;++i){
    revolution s_rev = sphere(C[i],R);
    surface s_surf = surface(s_rev);
    draw(s_surf, nlpen);
    draw(s_rev.silhouette(100), black+linewidth(3));
  }
}

// Draw a cylinder
void Draw(guide3 g,pen p=currentpen, real cylR=0.2){
  draw(
    cylinder(
      point(g,0),cylR,arclength(g),point(g,1)-point(g,0)
    ).surface(
               new pen(int i, real j){
                 return p;
               }
             )
  );
}

// Draw a cylinder without light
void Draw_nolight(guide3 g,pen p=currentpen, real cylR=0.2){
  material nlpen = material(diffusepen=opacity(1.0), emissivepen=p, shininess=0);
  revolution s_rev = cylinder(point(g,0),cylR,arclength(g),point(g,1)-point(g,0));
  surface s_surf = surface(s_rev);
  draw(s_surf, nlpen);
  draw(s_rev.silhouette(100), black+linewidth(3));
}

triple sphere1=(4.225, 0.0, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(4.225, 2.1125, 3.8805995447981757e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(2.1125, 2.1125, 2.5870663631987837e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(4.225, 0.0, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(4.225, 2.1125, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(2.1124999999999994, 4.225, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(2.1125, 2.1125, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(-1.2935331815993918e-16, 2.1125, 2.1125);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(2.1125, 0.0, 1.2935331815993918e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(-2.5870663631987837e-16, 4.225, 2.5870663631987837e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(2.1125, 0.0, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(2.1124999999999994, 4.225, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(4.225, 2.1125, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(2.1125, 2.1125, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(4.225, 4.225, 5.174132726397567e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(-2.5870663631987837e-16, 4.225, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(-2.5870663631987837e-16, 4.225, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(4.225, 4.225, 4.2250000000000005);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(0.0, 0.0, 0.0);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(2.1125, 0.0, 2.1125);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(4.225, 4.225, 2.1125000000000003);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(0.0, 0.0, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(-1.2935331815993918e-16, 2.1125, 1.2935331815993918e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(4.225, 0.0, 2.5870663631987837e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('8aff00')+opacity(1.0));

triple sphere1=(0.0, 0.0, 2.1125);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(-1.2935331815993918e-16, 2.1125, 4.225);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));

triple sphere1=(2.1124999999999994, 4.225, 3.8805995447981757e-16);


triple[] spheres = {sphere1};
drawSpheres(spheres, 0.5, rgb('ff0d0d')+opacity(1.0));
pen connectPen=rgb('000000') + linewidth();

triple IPOS = (0, 0, 0);
triple FPOS = (4.225, 0.0, 2.5870663631987837e-16);
draw(IPOS--FPOS, connectPen);

triple IPOS = (0, 0, 0);
triple FPOS = (-2.5870663631987837e-16, 4.225, 2.5870663631987837e-16);
draw(IPOS--FPOS, connectPen);

triple IPOS = (0, 0, 0);
triple FPOS = (0.0, 0.0, 4.225);
draw(IPOS--FPOS, connectPen);

triple IPOS = (4.225, 0.0, 2.5870663631987837e-16);
triple FPOS = (4.225, 4.225, 5.174132726397567e-16);
draw(IPOS--FPOS, connectPen);

triple IPOS = (4.225, 0.0, 2.5870663631987837e-16);
triple FPOS = (4.225, 0.0, 4.225);
draw(IPOS--FPOS, connectPen);

triple IPOS = (-2.5870663631987837e-16, 4.225, 2.5870663631987837e-16);
triple FPOS = (4.225, 4.225, 5.174132726397567e-16);
draw(IPOS--FPOS, connectPen);

triple IPOS = (-2.5870663631987837e-16, 4.225, 2.5870663631987837e-16);
triple FPOS = (-2.5870663631987837e-16, 4.225, 4.225);
draw(IPOS--FPOS, connectPen);

triple IPOS = (0.0, 0.0, 4.225);
triple FPOS = (4.225, 0.0, 4.225);
draw(IPOS--FPOS, connectPen);

triple IPOS = (0.0, 0.0, 4.225);
triple FPOS = (-2.5870663631987837e-16, 4.225, 4.225);
draw(IPOS--FPOS, connectPen);

triple IPOS = (4.225, 4.225, 5.174132726397567e-16);
triple FPOS = (4.225, 4.225, 4.2250000000000005);
draw(IPOS--FPOS, connectPen);

triple IPOS = (4.225, 0.0, 4.225);
triple FPOS = (4.225, 4.225, 4.2250000000000005);
draw(IPOS--FPOS, connectPen);

triple IPOS = (-2.5870663631987837e-16, 4.225, 4.225);
triple FPOS = (4.225, 4.225, 4.2250000000000005);
draw(IPOS--FPOS, connectPen);
