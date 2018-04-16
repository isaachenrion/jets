import logging
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

def get_emailer(email_filename):
    with open(email_filename, 'r') as f:
        lines = f.readlines()
        recipient, sender, password = (l.strip() for l in lines)
    return Emailer(sender, password, recipient)

class Emailer:
    '''
    Handles email. Given a sender, password and recipient, it can send messages
    with attachments over a server.
    '''
    def __init__(self, sender, password, recipient):
        self.sender = sender
        self.password = password
        self.recipient = recipient

    def send_msg(self, text, subject, attachments=None):
          msg = MIMEMultipart()
          msg['From'] = self.sender
          msg['To'] = self.recipient if self.recipient is not None else self.sender
          msg['Date'] = formatdate(localtime = True)
          msg['Subject'] = subject

          msg.attach(MIMEText(text))

          if attachments is not None:
              for f in attachments:
                  part = MIMEBase('application', "octet-stream")
                  part.set_payload( open(f,"rb").read() )
                  encoders.encode_base64(part)
                  part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(f)))
                  msg.attach(part)

          server = smtplib.SMTP('smtp.gmail.com:587')
          server.ehlo()
          server.starttls()
          server.login(self.sender, self.password)

          sent = False
          attempts = 0
          while not sent and attempts < 10:
              try:
                  server.sendmail(self.sender, self.recipient, msg.as_string())
                  sent = True
              except smtplib.SMTPException:
                  time.sleep(5)
                  attempts += 1

          server.close()
          logging.info("SENT EMAIL")
