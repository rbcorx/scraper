import scrapy
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk
import re

st = StanfordNERTagger("/Users/rbcorx/stalford_nlp/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz", encoding="utf-8")

"""
check notes file



export CLASSPATH=/Users/rbcorx/stalford_nlp/stanford-ner-2016-10-31:/Users/rbcorx/stalford_nlp/stanford-postagger-full-2016-10-31
export STANFORD_MODELS=/Users/rbcorx/stalford_nlp/stanford-postagger-full-2016-10-31/models:/Users/rbcorx/stalford_nlp/stanford-ner-2016-10-31/classifiers:/Users/rbcorx/stalford_nlp

model (3 class)
/Users/rbcorx/stalford_nlp/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz

tokenizer:
/Users/rbcorx/stalford_nlp/stanford-postagger-full-2016-10-31/stanford-postagger.jar

ALGO: out of classified entities, find entities in the same text window and join them

Note: watch out for scrapy utf-8 encoding

ALGO2: fetch keywords, perform only proximity search over keywords for named entities
    Identify areas of high interest first and perform NER in these areas exclusively


def extract_entities(text):
     for sent in nltk.sent_tokenize(text):
         for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
             if hasattr(chunk, 'node'):
                 print chunk.node, ' '.join(c[0] for c in chunk.leaves())

"""


def extract_entities(text):
     print "sldfsdf@@@@@@@@@@@@@@@@"
     for sent in nltk.sent_tokenize(text):
         for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
             if hasattr(chunk, 'node'):
                print chunk.node, ' '.join(c[0] for c in chunk.leaves())
                if chunk.node == "PERSON":
                    name = ' '.join(c[0] for c in chunk.leaves())
                    print chunk.node, name
                    return name


caps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lower = 'abcdefghijklmnopqrstuvwxyz'


class TestSpider(scrapy.Spider):
    name = "test"



    xq_a_text = '//a[contains(translate(text(), \'{}\', \'{}\'),'.format(caps, lower) + ' \'{}\')]/@href'

    contacts_pages = [
        "leadership",
        "management",
        "team",
        "company",
        "we are",
        "about",
        "story", # our
    ]
    careers_pages = [
        "careers",
        "work", # TODO too generic, needs more qualifiers!
        "join", # TODO too generic, needs more qualifiers!
    ]
    careers_tags_deps = [
        "technology",
        "engineering",
        "product",
        "development",
        "software",
    ]

    jobs_tags = [
        "ruby",
        "python",
        "full stack",
        "full-stack",
        "software",
        "engineer",

        "front end",
        "front-end",
        "frontend",

        "backend",
        "back end",
        "back-end",

    ]
    # TODO add: ["founder", "engineering"] ["founder", "technology"]
    poi = [
        ["ceo", "chief executive officer", "president", "chairman", "founder"],
        ["cto", "chief technology officer", "chief technical officer",
         ["director", "engineering"],
         ["director", "technology"],
         ["vp", "engineering"],
         ["vp", "technology"],
         ["svp", "technology"],
         ["svp", "engineering"]],
        ["cfo", "chief finance officer", "chief financial officer", "finance", "financial"],
    ]

    titles = [
        "vp",
        "svp",
        "director",
        "sales",
        "expert",
        "assistant",
        "head",
        "hr",
        "chief",
        "executive",
        "officer",
        "president",
        "chairman",
        "vice",
    ]
    partial_titles = [
        "recruit",
        "market",
        "sales",
    ]




    def start_requests(self):
        """
        Alternative:
        start_urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        """
        urls = [
            # "https://www.fyber.com/",
            # "http://inmobi.com/",
            # "http://www.appsflyer.com",
            # "http://startuphyderabad.com/",
            # "http://science37.com/",  # Problem, js. requires phantomjs or selenium
            # "http://siftrock.com/",
            # "http://moat.com/",
            # "http://shyp.com/",
            # "http://clever.com/",
            #"https://influential.co/",
            "http://site.adform.com/",
        ]
        self.best_res = [[] for i in range(3)]
        self.matches = {}
        self.url = None

        for url in urls:
            # purging state
            # TODO better state encapsulation
            # for i in range(len(TestSpider.best_res)):
            #     del TestSpider.best_res[i][:]
            # TestSpider.matches.clear()
            # print "clearing!!!"
            self.best_res = [[] for i in range(3)]
            self.matches = {}
            self.url = url
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        skip_rest = False
        for pages, parser in [(TestSpider.contacts_pages, self.parse_contacts), (TestSpider.careers_pages, self.parse_careers)]:
            done = False
            for page_term in pages:
                if skip_rest:
                    if done:
                        break
                # import ipdb
                # ipdb.set_trace()
                # print "\n\n ::: SEARCHING ::: \n\n"
                prospect = response.xpath(TestSpider.xq_a_text.format(page_term)).extract()
                if prospect is not None:
                    # print "\n\n ::: GOT PROSPECT ::: \n\n"
                    for current in prospect:
                        # if len(response.xpath("//a[contains(@href, '{}') and contains(text(), 'about')]/text()".format(current)).extract_first()) > 20:
                        #     # avoiding links with longer texts
                        #     continue
                        if "press" in current or "news" in current:
                            continue
                        done = True
                        yield response.follow(current, parser)


        # page = response.url.split("/")[-2]
        # filename = 'test-%s.html' % page
        # with open(filename, 'wb') as f:
        #     print response.body[:100]
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)

    def parse_contacts(self, response):
        self.best_res = [[] for i in range(3)]
        self.matches = {}
        verbose = False
        """
        BUGS: {
            # TODO: not detecting naveen tiwari from inmobi
            # TODO: not detecting appsflyer CTO Reshef Mann
            # TODO: detects North America as a person and not location
        }
        TODO: check for links in contacts pages too!"""
        # TODO tranfrom to binary search
        # TODO narrow search, limited team members, top contacts only?
        # TODO precedence of page headers: leadership, about us : in order
        # TODO leadership link inside this page?
        # TODO extract meta tags for succesful extractions to keep a track of data features for using in ML in the future
        # Finding CEO, extract more tags/bio info to build better distance algorithm
        """
        giant Search blob more efficient? find classes by bin edges
        CEO = ["CEO", "Chief ...", "President", "Chairman", ]

        Chief/Director/SVP/VP of Engineering/Product/Technology
        CTO
        Finance
        # TODO level 1/level 2 weighted annotations
        """
        def rank_in_class(annotation, match, find_first=False):
            # TODO rank by match length too
            # TODO distance between class titles and match
            ranks = []
            for ic, cclass in enumerate(TestSpider.poi):
                ranks.append(0)

                for rank, key in enumerate(cclass):
                    if type(key) is list:
                        key = r"\b{}\b".format((r"\b.+\b").join([k.encode("ascii", "replace") for k in key]))
                    else: key = r"\b{}\b".format(key.encode("ascii", "replace"))

                    rank_matches = [m.start() for m in re.finditer(key, annotation, re.IGNORECASE)]
                    ranks[ic] += len(rank_matches) *\
                        (10 / len(cclass)) * (len(cclass) - rank) * 100
                    if rank_matches:
                        # TODO For multi keyword, consider including length of match too in teh the heuristic as
                        #       eg: vp sdlfksdfljsdfljsdflk engineering means nothing for a dist vector
                        match_i = annotation.find(match)
                        vector_dist = 0
                        if match_i > -1:
                            dist = abs(rank_matches[0] - match_i)
                            if dist < 20:
                                vector_dist = (len(annotation) - dist) * (10 / len(cclass)) * (len(cclass) - rank) * 100
                        ranks[ic] += vector_dist
                    if find_first and ranks[ic] > 0:
                        break

            ranks = [rank * len(match.split()) for rank in ranks]
            # best_rank = max(ranks)
            # if best_rank == 0 and annotation.lower().find("ceo") >= 0:
            #     print "damn fucked!!!!"
            # return (ranks.index(best_rank), best_rank * len(match.split()))
            return ranks



        def expand_search(tokenized, ind, entity_inds, check_dom=True):
            """
            DOM search: names should be in the same DOM element
            """
            # if tokenized[ind] in TestSpider.name_filters:
            #     return []
            def filter_name(name):
                low = name.lower()
                for title in TestSpider.titles:
                    if title == low:
                        return False
                for title in TestSpider.partial_titles:
                    if title in low:
                        return False
                return True

            name = [tokenized[ind], ]
            if not filter_name(name[0]):
                return ("", 0)

            l = ind-1 if ind-1 > 0 else None
            r = ind+1 if ind+1 < len(tokenized) else None
            while l is not None or r is not None:

                if l is not None and filter_name(tokenized[l]) and tokenized[l] in entity_inds:  # and tokenized[l] not in TestSpider.name_filters
                    if response.xpath("//*[contains(text(), '{}')]".format((tokenized[l] + " " + " ".join(name)).encode("ascii", "replace"))):
                        name.insert(0, tokenized[l])
                        l = l-1 if l-1 > 0 else None
                    else: l = None
                else: l = None

                if r is not None and filter_name(tokenized[r]) and tokenized[r] in entity_inds:  # and tokenized[r] not in TestSpider.name_filters
                    if response.xpath("//*[contains(text(), '{}')]".format((" ".join(name) + " " + tokenized[r]).encode("ascii", "replace"))):
                        name.append(tokenized[r])
                        r = r+1 if r+1 < len(tokenized) else None
                    else: r = None
                else: r = None
            return (" ".join(name), len(name))



        print "\nExtracting text for prospecting ::: \n"
        text = " ".join(response.xpath("//body//text()").extract()).strip()
        tokenized = word_tokenize(text)
        # print tokenized
        entities = list(set(map(lambda x: x[0], filter(lambda x: x[1] == 'PERSON', st.tag(tokenized)))))
        entity_inds = {}
        # print entities
        for entity in entities:
            if entity not in entity_inds:
                entity_inds[entity] = [i for i, el in enumerate(tokenized) if el == entity]

        matches = []
        for entity in entity_inds:
            for ind in entity_inds[entity]:
                name = expand_search(tokenized, ind, entity_inds)
                if name[1] > 0 and len(name[0]) > 2:
                    # print "found multi worded name!!! "
                    # print name[0]
                    matches.append(name[0])

        # if sirnames were detected, abandon all single worded names
        with_sirnames = list(filter(lambda x: x.find(" ") >= 0, matches))
        if with_sirnames:
            matches = with_sirnames

        # match filters:
        # removed = {}
        # for match1 in matches:
        #     if match1 in removed:
        #         continue
        #     for match2 in matches:
        #         if match2 in removed:
        #             continue
        #         if match1 == match2:
        #             continue
        #         if len(match1) > len(match2):
        #             hay = match1
        #             needle = match2
        #         else:
        #             hay = match2
        #             needle = match1
        #         if hay.find(needle) >= 0:
        #             removed[hay] = True
        # filtered = [match.encode("ascii", "replace") for match in matches if match not in removed]

        matches = {match.encode("ascii", "replace"): [] for match in matches}  # filtered}

        for annote in ["//*[contains(text(),'{}')]//text()", "//*[contains(text(),'{}')]/parent::*//text()"]:
            for match in matches:
                annotations = (" ".join([" ".join(x.split()) for x in response.xpath(annote.format(match)).extract()]))
                # annotations.replace(match, "")
                matches[match].append(annotations)

        # print matches
        # import ipdb
        # ipdb.set_trace()

        best_in_class = [[] for _ in TestSpider.poi]

        for i, match in enumerate(matches.keys()):
            if verbose:
                print "\n{}. {}".format(i, match)
            # print "\nANNOTATION LEV-1:"
            # print matches[match][0][:150]
            # print "\nANNOTATION LEV-2 (BROADER):"
            # print matches[match][1][:150]
            ranks = rank_in_class(matches[match][1], match)
            if verbose:
                print "match: {}; ranks : {}".format(match, ranks)

            # print best_in_class
            # print rank
            for i, rank in enumerate(ranks):
                best_in_class[i].append((match, rank))
            # if best_in_class[rank[0]][1] < rank[1]:
            #     # TODO handle equality condition in ranks

            #     best_in_class[rank[0]] = (match, rank[1])

        # print "@@@@@@ Best class matches found were :: @@@@@@@"
        for i, clas in enumerate(best_in_class):
            if clas:
                best_so_far = self.best_res[i][0] if len(self.best_res[i]) else None
                self.best_res[i].extend(clas)
                self.best_res[i].sort(key=lambda x: x[1], reverse=True)
                if (not best_so_far) or (self.best_res[i][0] != best_so_far):
                    pass
                    # import ipdb
                    # ipdb.set_trace()
                    # print "\n{}. {}, found to be in: {}\n".format(i, self.best_res[i][0][0], TestSpider.poi[i])
                    # print matches[self.best_res[i][0][0]][-1][:150]
            else:
                print "no match for class!!!"
                # TODO if a page has too many of these, the page might be a wrong one, like the press kit page at "http://clever.com/"

        n_top_disp = 3

        # self.matches.update(matches)
        for iclass, candidates in enumerate(self.best_res):
            print "\n\nClass Results: {}\n".format(TestSpider.poi[iclass])
            for i, cand in enumerate(candidates):
                if i+1 > n_top_disp:
                    break
                print "\n{}. {}. RANK: {}\n".format(i, cand[0], cand[1])
                print matches[cand[0]][-1][:300], "\n"



        # yield {"extracted": matches}

    def parse_careers(self, response):
        # TODO js dynaimic positions loading handle with selenium
        pass

    def parse_2(self, response):
        """use feed exports while running to save all quotes to json"""
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').extract_first(),
                'author': quote.css('small.author::text').extract_first(),
                'tags': quote.css('div.tags a.tag::text').extract(),
            }
    def parse_3(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').extract_first(),
                'author': quote.css('small.author::text').extract_first(),
                'tags': quote.css('div.tags a.tag::text').extract(),
            }

        next_page = response.css('li.next a::attr(href)').extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
            # shortcur
            # no need for urljoin()
            # yield response.follow(next_page, callback=self.parse)
            # shorter-cut
            # for href in response.css('li.next a')
            #   yield response.follow(href, callback=self.parse)





class AuthorSpider(scrapy.Spider):
    name = 'author'

    start_urls = ['http://quotes.toscrape.com/']

    def parse(self, response):
        # follow links to author pages
        for href in response.css('.author + a::attr(href)'):
            yield response.follow(href, self.parse_author)

        # follow pagination links
        for href in response.css('li.next a::attr(href)'):
            yield response.follow(href, self.parse)

    def parse_author(self, response):
        def extract_with_css(query):
            return response.css(query).extract_first().strip()

        yield {
            'name': extract_with_css('h3.author-title::text'),
            'birthdate': extract_with_css('.author-born-date::text'),
            'bio': extract_with_css('.author-description::text'),
        }



if __name__ == "__main__":
    pass

