import collections
import csv
import re


def _csvdictline_to_string(dictline, fieldnames, delimiter, quotechar):
    linestring = ''
    for fn in fieldnames:
        if delimiter in dictline[fn]:
            linestring += '{0}{1}{0}'.format(quotechar, dictline[fn]) + delimiter
        else:
            linestring += dictline[fn] + delimiter
    return linestring[:-len(delimiter)]


def filter(csvfile, include={}, exclude={}, fieldnames=None, delimiter=',', quotechar='"'):
    """
    Filters a csv file to include only items containing the correct values in the columns.

    Args:
        csv: The csv file to be filtered.
             Must be a list, file, or file-like object that supports the iterator protocol that returns strings as
             values.
             A column from the csv that contains strings will be considered a list delimited by the same delimiter as
             the columns themselves.
        include: A dictionary with a mapping from csv-header to a list or set of allowed values in the filtered csv.
                 In a column with lists as values, any 1 of the values from the list can match any of the expected
                 values from include.
        exclude: Works oppositely to include. If any value from the csv in the applicable header matches, exclude that
                 item from the returned list.
                 Note that exclude is stronger than include and will remove items that should otherwise have been
                 included.
        fieldnames: A list of headers, to be used if the csv does not have headers of their own.
                    If None, the first row of the csv is presumed to be the header columns.
        delimiter: The delimiter to be used to separate values in the csv.
        quotechar: The character which indicates the start and end of strings (lists).

    Returns:
        A list of strings from csvfile that passed the filters.

    Note:
        Any column that are not present in either include or exclude is going to be ignored and will be included
        blindly in the output list.

    Example usage:
        a_csv = ['a,b', '1,0', '"2,0",0', '5,5', '1,1']
        ret = filter(a_csv, include={'a': {'1', '0'}}, exclude={'b': {'1'}})
        # ret is equal to ['a,b, '1,0', '"2,0",0']
    """
    def filter_line(line, constraints):
        filters = []
        for constraint in constraints:
            constraint_line = line[constraint].split(',')
            constraint_line = [str(value) in constraints[constraint] for value in constraint_line]
            filters.append(any(constraint_line))
        return filters

    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
    filtered_csv = []

    # DictReader removes headers from the read file if we did not supply our own
    # If so, re-add the headers.
    include_headers = not fieldnames
    fieldnames = csvreader.fieldnames

    if include_headers:
        linestring = ''
        for fn in fieldnames:
            if delimiter in fn:
                linestring += '{0}{1}{0}'.format(quotechar, fn) + delimiter
            else:
                linestring += fn + delimiter
        filtered_csv.append(linestring[:-len(delimiter)])

    for line in csvreader:
        if any(filter_line(line, exclude)):
            continue
        if sum(filter_line(line, include)) != len(include):
            continue

        filtered_csv.append(_csvdictline_to_string(line, fieldnames, delimiter, quotechar))
    return filtered_csv


def cut(csvfile, columns, fieldnames=None, delimiter=',', quotechar='"'):
    """
    Creates a new csvlist containing only the specified columns.
    """
    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)

    include_headers = not fieldnames
    fieldnames = csvreader.fieldnames
    fieldnames = [fieldname for fieldname in fieldnames if fieldname in columns]

    filtered_csv = []
    if include_headers:
        linestring = ''
        for fn in fieldnames:
            if delimiter in fn:
                linestring += '{0}{1}{0}'.format(quotechar, fn) + delimiter
            else:
                linestring += fn + delimiter
        filtered_csv.append(linestring[:-len(delimiter)])

    for line in csvreader:
        filtered_csv.append(_csvdictline_to_string(line, fieldnames, delimiter, quotechar))
    return filtered_csv


def csvtolist(csvfile, delimiter=',', quotechar='"'):
    """
    Translates a CSV iterator into a list of strings.
    The list of strings can be used in any of the other CSV util functions as well as the csv lib.

    Args:
        csvfile: The csv file to be converted.
                 Must be a list, file, or file-like object that supports the iterator protocol that
                 returns strings as values.
                 A column from the csv that contains strings will be considered a list delimited by
                 the same delimiter as the columns themselves.
        delimiter: The delimiter to be used to separate values in the csv.
        quotechar: The character which indicates the start and end of strings (lists).

    Returns:
        A list of strings representing the csv.
    """
    csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
    new_list = []
    for line in csvreader:
        l = ''
        for item in line:
            if delimiter in item:
                l += '{0}{1}{0}'.format(quotechar, item) + delimiter
            else:
                l += item + delimiter
        new_list.append(l)
    return new_list


def csvtodict(csvfile, key, split_strings=True, fieldnames=None, delimiter=',', quotechar='"'):
    """
    Translates a CSV into a dictionary using the values in one of the columns as the keys for the dictionary.

    Args:
        csvfile: The csv file to be converted.
                 Must be a list, file, or file-like object that supports the iterator protocol that
                 returns strings as values.
                 A column from the csv that contains strings will be considered a list delimited by
                 the same delimiter as the columns themselves.
        key: The column from which the dict keys are to be collected from.
             `key` has to be present in the dictionary headers as defined either by `fieldnames` or the first line
             of the csv.
        split_strings: If True, splits strings on the delimiter for the file into lists.
                       Does not affect the keys of the returned dictionary, just the values.
        fieldnames: A list of headers, to be used if the csv does not have headers of their own.
                    If None, the first row of the csv is presumed to be the header columns.
        delimiter: The delimiter to be used to separate values in the csv.
        quotechar: The character which indicates the start and end of strings (lists).

    Returns:
        A dict of dicts representing the csv.

    Note:
        The returned dictionary is unordered.
        Each "row" will include its own key in addition to all other items.

    Example usage:
        a_csv = ['a,b', '1,0', '"2,0",0']
        ret = csvtodict(a_csv, 'a')
        # ret is equal to {'1': {'a': '1', b: '0'}, '"2,0"': {'a': '"2,0"', 'b': '0'}}
    """
    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
    translated = {}

    for line in csvreader:
        translated[line[key]] = {}
        for header in line:
            s = line[header]
            if split_strings and delimiter in s:
                s = s.split(delimiter)
            translated[line[key]][header] = s
    return translated


def grep(csvfile, header, match, match_type='exact', split_strings=True,
         fieldnames=None, delimiter=',', quotechar='"'):
    """
    Args:
        csv: The csv file to be searched.
             Must be a list, file, or file-like object that supports the iterator protocol that returns strings as
             values.
             A column from the csv that contains strings will be considered a list delimited by the same delimiter
        header: Which column to search.
                If `fieldnames` is None, `header` must exist in the first row of the csv.
                If `fieldnames` is set, `header` must instead exist within that list.
        match: What to match against.
               If `match` is a list, the item can match either of the items in the list.
               List matching is very very slow for large lists.
        match_type: Can be one of 'exact', 'partial', 'prefix', 'suffix'.
                    'exact': the lines value must be exactly equal to `match` in the specified column.
                    'partial': It is enough for `match` to exist within the specified column.
                    'prefix': The value in the column must start with `match`.
                    'suffix': The value in the column must end with `match`.
        split_strings: If True, splits strings on the delimiter for the file into lists.
                       In this mode, only ONE of the items in the list has to match the search terms.
        fieldnames: A list of headers, to be used if the csv does not have headers of their own.
                    If None, the first row of the csv is presumed to be the header columns.
        delimiter: The delimiter to be used to separate values in the csv.
        quotechar: The character which indicates the start and end of strings (lists).

    Returns:
        A list of strings from csvfile that all match the specified search.
    """
    patterns = []
    if not isinstance(match, collections.Iterable) and not isinstance(match, str):
        match = [match]
    for m in match:
        m = re.escape(m)
        if match_type == 'exact':
            patterns.append(re.compile('^{}$'.format(m)))
        elif match_type == 'partial':
            patterns.append(re.compile('.*?{}.*?'.format(m)))
        elif match_type == 'prefix':
            patterns.append(re.compile('^{}'.format(m)))
        elif match_type == 'suffix':
            patterns.append(re.compile('.*{}$'.format(m)))
        else:
            raise ValueError('match_type must be one of exact, partial, prefix, or suffix')

    csvreader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)

    filtered_csv = []
    include_headers = not fieldnames
    fieldnames = csvreader.fieldnames

    if include_headers:
        linestring = ''
        for fn in fieldnames:
            if delimiter in fn:
                linestring += '{0}{1}{0}'.format(quotechar, fn) + delimiter
            else:
                linestring += fn + delimiter
        filtered_csv.append(linestring[:-len(delimiter)])

    for line in csvreader:
        s = line[header]
        if split_strings:
            s.split(delimiter)
        else:
            s = [s]

        found = False
        for ss in s:
            if any(map(lambda pattern: pattern.match(ss), patterns)):
                found = True
                break

        if found:
            filtered_csv.append(_csvdictline_to_string(line, fieldnames, delimiter, quotechar))

    return filtered_csv
