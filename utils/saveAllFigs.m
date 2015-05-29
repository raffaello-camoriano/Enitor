h = get(0,'children');
for i=1:length(h)
  saveas(h(i), [figsdir 'figure' num2str(i) '.fig']);
  saveas(h(i), [figsdir 'figure' num2str(i) '.png']);
  saveas(h(i), [figsdir 'figure' num2str(i) '.eps']);
end