%-Carres-

include "globals.mzn";

array[1..5] of var 10..100: x;

% Les entiers sont différents l'un de l'autre.
constraint alldifferent(x);
% La somme des carrés vaut le carré d'un autre entier
constraint pow(x[1],2) + pow(x[2],2) + pow(x[3],2) + pow(x[4],2) == pow(x[5],2);

solve minimize x[5];

output [
  show(x)
];