function gph = newGphA(Pt, parGph)
% Generate a graph by connecting points.
%
% Remark
%   The edge feature is asymmetric.
%
% Input
%   Pt      -  graph node, d x n
%   parGph  -  parameter for computing the adjacency matrix
%              see gphEg.m for more details
%
% Output
%   gph     -  graph
%     Pt    -  graph node, d x n
%     Eg    -  graph edge, 2 x m
%     vis   -  binary indicator of nodes that have been kept, 1 x n | []
%     G     -  node-edge adjacency (for starting point), n x m
%     H     -  node-edge adjacency (for ending point), n x m
%     PtD   -  edge feature, 2 x m
%     dsts  -  distance, 1 x m
%     angs  -  angle, 1 x m
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 08-11-2011
%   modify  -  Feng Zhou (zhfe99@gmail.com), 03-06-2013

% dimension
n = size(Pt, 2);

% edge
[Eg, vis] = gphEg(Pt, parGph);

% incidence for asymmetric edge
[G, H] = gphEg2IncA(Eg, n);

% second-order feature
[PtD, dsts, angs, angAs] = gphEg2Feat(Pt, Eg);

% store
gph.Pt = Pt;
gph.Eg = Eg;
gph.vis = vis;
gph.G = G;
gph.H = H;
gph.PtD = PtD;
gph.dsts = dsts;
gph.angs = angs;
gph.angAs = angAs;
