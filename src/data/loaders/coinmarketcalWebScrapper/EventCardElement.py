from selenium.webdriver.remote.webelement import WebElement

from parse import parse
import re
from datetime import datetime, date

class EventCardElement(WebElement):

    def __init__(self, webElement: WebElement):
        super().__init__(webElement._parent, webElement._id, webElement._w3c)

    @property
    def added_date(self) -> date:
        added_date_text = self.find_element_by_class_name('added-date').text
        added_date_text = re.findall('\d{2} \w{3} \d{4}', added_date_text)[0]
        return datetime.strptime(added_date_text,'%d %b %Y').date()

    @property
    def event_date(self) -> date:
        added_date_text = self.find_element_by_class_name('card__date').text
        added_date_text = re.findall('\d{2} \w{3} \d{4}', added_date_text)[0]
        return datetime.strptime(added_date_text,'%d %b %Y').date()

    @property
    def votes(self) -> int:
        votes_txt = self.find_element_by_class_name('progress__votes').text
        matches = re.findall('\d+', votes_txt)
        return int(matches[0]) if matches else 0
    
    @property
    def confidence_percentage(self) -> float:
        # percentage_text = self.find_element_by_class_name('progress__percent').text
        percentage_text = self.find_element_by_class_name('progress-bar').get_attribute('aria-valuenow')
        return float(percentage_text)/100.0 if percentage_text else float('nan')
    
    @property
    def title(self) -> str:
        return self.find_element_by_class_name('card__title').text

    @property
    def description(self) -> str:
        return self.find_element_by_class_name('card__description').text.strip('"')