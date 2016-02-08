h = get(0,'children');
for i=1:length(h)
  saveas(h(i), [figsdir 'figure' num2str(i) '.fig']);
  pause(0.1);
  saveas(h(i), [figsdir 'figure' num2str(i) '.png']);
  pause(0.1);
  saveas(h(i), [figsdir 'figure' num2str(i) '.eps']);
  pause(0.1);
end