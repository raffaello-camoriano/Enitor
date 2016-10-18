function playGong()
%PLAYGONG Play Gong sound

    load gong;
%     load chirp;
%     load handel;
    player = audioplayer(y, Fs);
    playblocking(player);
end

