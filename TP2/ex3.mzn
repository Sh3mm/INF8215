%-Attribution de taches-

include "globals.mzn";

array[1..N,1..S] of int: cost;%cout d'une tache dependamment du sous-traitant
array[1..N,1..S] of int: duration;%duree d'une tache dependamment du sous-traitant, en jours

int: S;%nombre de sous-traitants
int: N;%nombre de tâches
int: T;%horizon temporel, en jours

array[1..N] of var 1..S: attribution;
constraint forall(i in 2..N)(attribution[i] != attribution[1]);
constraint forall(i in 3..N)(attribution[i] != attribution[2]);

% var int: C = <A COMPLETER>;

% <A COMPLETER>
%solve minimize C;

output [
  "x: ", show(x), "\n",
  "Cost: ", show(C)]
;