import logging
import os

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
class Emailer:
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
          server.sendmail(self.sender, self.recipient, msg.as_string())
          server.close()
          logging.info("SENT EMAIL")
