%-Enigme des mariages-

include "globals.mzn"; 

int: n = 5;

int: Dupont = 1;
int: Tremblay = 2;
int: Durand = 3;
int: Gagnon = 4;
int: Petit = 5;

var 1..n: Remy;
var 1..n: David;
var 1..n: Charles;
var 1..n: Pierre;
var 1..n: Edouard;
array[1..n] of var 1..n: M = [Remy, David, Charles, Pierre, Edouard];
 
var 1..n: Emilie;
var 1..n: Juliette;
var 1..n: Maria;
var 1..n: Elisabeth;
var 1..n: Manon;
array[1..n] of var 1..n: F = [Emilie, Juliette, Maria, Elisabeth, Manon];

array[1..n] of var 0..100: married;

% Juliette n’est pas mariée depuis aussi longtemps que Charles ou les Tremblays, mais depuis plus longtemps que David ou les Duponts.
constraint married[Juliette] < married[Charles];
constraint married[Juliette] < married[Tremblay];
constraint married[Juliette] > married[David];
constraint married[Juliette] > married[Dupont];
% Élisabeth est mariée depuis deux fois plus longtemps que les Durands, mais deux fois moins longtemps que Rémy.
constraint married[Elisabeth] == 2 * married[Durand];
constraint married[Elisabeth] * 2 == married[Remy];
% Les Petits sont mariés depuis 10 ans de plus que Pierre et 10 ans de moins que Maria.
constraint married[Petit] == married[Pierre] + 10;
constraint married[Petit] == married[Maria] - 10;
% David et Manon sont mariés depuis 25 ans demoins que les Gagnons qui, mariés depuis 30 ans, sont le couple marié depuis le plus longtemps.
constraint David == Manon;
constraint married[David] == married[Gagnon] - 25;
constraint married[Gagnon] == 30;
constraint max(married) == married[Gagnon];
% Ni Émilie ni les Duponts ont été mariés depuis le moins longtemps.
constraint married[Emilie] != min(married);
constraint married[Dupont] != min(married);
% Édouard est marié depuis 25 ans.
constraint married[Edouard] == 25;

solve satisfy;

output [
  "M: ", show(M), "\n",
  "F: ", show(F), "\n",
  "married : ", show(married), "\n",
]
;

% M: [3, 1, 2, 1, 2]
% F: [1, 5, 2, 3, 1]
% married : [5, 25, 0, 30, 15]