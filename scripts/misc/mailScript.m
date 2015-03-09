% Modify these two lines to reflect
% your account and password.
myaddress = 'raffaello2@libero.it';
mypassword = 'provaprova90';

setpref('Internet','E_mail',myaddress);
setpref('Internet','SMTP_Server','smtp.libero.it');
setpref('Internet','SMTP_Username',myaddress);
setpref('Internet','SMTP_Password',mypassword);

props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', ...
                  'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');

sendmail('raffaello2@gmail.com', 'Gmail Test', 'This is a test message.');