h = get(0,'children');
for saveAllFigsIdx=1:length(h)
  saveas(h(saveAllFigsIdx), [figsdir 'figure' num2str(saveAllFigsIdx) '.fig']);
  pause(0.1);
  saveas(h(saveAllFigsIdx), [figsdir 'figure' num2str(saveAllFigsIdx) '.png']);
  pause(0.1);
  saveas(h(saveAllFigsIdx), [figsdir 'figure' num2str(saveAllFigsIdx) '.eps']);
  pause(0.1);
end