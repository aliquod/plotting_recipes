import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import textwrap
import seaborn as sns
from colour import Color
from copy import deepcopy
from inflect import engine
from math import ceil
from collections import deque

class Category:
    '''Main class of visualizing treemaps, representing a section of the map.
    Categories can split into subcategories, represented as a tree-like data structure.'''

    def __init__(self, name, value_name: str="",
                 supcategory=None, subcategories=[],
                 data=pd.DataFrame(),
                 is_root: bool=False,
                 is_others: bool=False) -> None:
        '''Creates a new category.
        [name] is the name of the category.
        [value name] refers to the name of the column that contains values whose proportions are represented as areas in the treemap; if left as the empty string, defaults to the number of observations under this category.
        [supcategory] is another Category object, should be a more general category in a multi-leveled map.
        [subcategories] is a list of Category objects, i.e. the finer categories under the same umbrella
        [data] is a pandas DataFrame object, containing data to be visualized. Rows should correspond to individual observations, columns containing the different specs.
        [is_root] is a boolean representing whether the category is the trivial contains-all category.
        [is_others] is a boolean representing whether the category is a temporary "others" category generated during visualization '''

        self.name = name

        # computing the [size] of the category, i.e. the count 
        if not value_name:
            # if not specified, default to counting the number of rows in the category.
            self.value_name = None
            self.size = data.shape[0]
        else:
            self.value_name = value_name
            self.size = data[value_name].sum()
        self.supcategory = supcategory
        self.subcategories = subcategories
        self.data = data
        self.is_root = is_root
        self.is_others = is_others
        self.grids = {}  # contains names of categories and their position in the tree map as a key:value pair

    @property
    def subcategories_in_descending_size(self):
        return sorted(self.subcategories, key=lambda category: category.size, reverse=True)

    def categories_from_df(self, df, group_by):
        '''Creates categories from a DataFrame (e.g. self.data), grouped by different values in the [group_by] columns'''

        category_dfs = {category_name: sub_df for category_name, sub_df in df.groupby(group_by)}
        subcategories = [Category(name=name, value_name=self.value_name, data=df) for name, df in category_dfs.items()]
        return subcategories

    def generate_subcategories(self, group_by):
        if not isinstance(group_by, list):
            group_by = [group_by]
        self.group_by = group_by

        subcategories = self.categories_from_df(self.data, group_by[0])

        # recursively generate finer subcategories for each subcategory
        if len(group_by) > 1:
            for subcategory in subcategories:
                subcategory.generate_subcategories(group_by = group_by[1:])
        self.add_subcategories(subcategories)

    def add_subcategories(self, subcategories):
        self.subcategories = subcategories
        for subcategory in subcategories:
            subcategory.supcategory = self

    def box_vis(self, img, configs, font_func):
        '''Creates a simple box-like visualization of itself, with color and sizes specified in [configs].'''

        # find the appropriate font size
        text = configs.label_generator(self.label)
        spliced_text, longest_line = StringHelper.split_lines(text, configs, font_func)
        font_size = StringHelper.largest_fitting_font_size(longest_line, font_func,
                                                       max_width = configs.width - 6*configs.line_width,
                                                       max_height = configs.height - 2*configs.line_width,
                                                       max_font_size = configs.base_font_size,
                                                       min_font_size = configs.min_font_size)

        # draw a box & write text
        ImageDraw.Draw(img).text(((configs.width//2, configs.height//2)), spliced_text, fill=configs.text_color, anchor='mm', font=font_func(font_size))
        ImageDraw.Draw(img).rectangle([(0,0), (configs.width , configs.height)],
                                        outline = configs.line_color,
                                        width = configs.line_width)
        return img

    def visualize(self, configs):
        '''Main drawing function, calls visualizations for the main map, creates the legend, then combines the two images'''

        main_map = self.visualize_main_map(configs)
        main_img = main_map["image"]
        color_dict = main_map["color_legend"]
        if configs.legend:
            # create legend
            legend = Legend(color_dict = color_dict)
            legend_img = legend.visualize(configs)
            # create larger canvas
            total_size = (configs.total_size(legend_img.size))
            img = Image.new('RGB', total_size, color = configs.top_color)
            # paste main map and legend
            img.paste(main_img, box=configs.main_map_ul)
            img.paste(legend_img, box=configs.legend_ul)
            return img
        return main_img

    def visualize_main_map(self, configs):
        '''Recursively draws the main map -- asks each subcategory to visualize itself, then pasting the images together.'''
        
        pil_font = lambda size: ImageFont.FreeTypeFont(font=configs.font, size=size)
        color_style = configs.top_level_color_style

        # if there are no more subcategories, just return the basic box-like visualization
        if len(self.subcategories) == 0:
            img = Image.new('RGB', (configs.width, configs.height), color = color_style.base_hex)
            return {"image":self.box_vis(img, configs, pil_font), "legend":{}}
        
        # otherwise create a new canvas
        img = Image.new('RGB', (configs.width, configs.height), color = color_style.base_hex)
        self.compute_grid(configs.width, configs.height)

        # compute cutoffs and merge subcategories that don't make it into an "other" category
        if configs.proportion_cutoff is None:
            for subcategory in self.subcategories_in_descending_size:
                sub_grid = self.grids[subcategory.name]
                if sub_grid["width"] == 0 or sub_grid["height"] == 0:
                    proportion_cutoff = subcategory.proportion_in_supcategory
                    break
            else:
                proportion_cutoff = 0
        else:
            proportion_cutoff = configs.proportion_cutoff
        not_cutoff_categories = [deepcopy(subcategory) for subcategory in self.subcategories if subcategory.proportion_in_supcategory > proportion_cutoff]
        not_cutoff_categories.sort(key=lambda cat: cat.size, reverse=True)
        cutoff_categories = [subcategory for subcategory in self.subcategories if subcategory.proportion_in_supcategory <= proportion_cutoff]

        if len(cutoff_categories) > 0:
            others_data = [category.data for category in cutoff_categories]
            others_category = Category(name="Other " + engine().plural(self.group_by[0]),
                                       value_name=self.value_name, supcategory=self, data=pd.concat(others_data),
                                       is_others = True)
            not_cutoff_categories.append(others_category)
        
        # create a temporary category holding the subcategories larger than the cutoff, as well as the merged "Others" category.
        temp_category = Category(name = self.name, value_name = self.value_name,
                                 supcategory = self.supcategory,
                                 data = self.data,
                                 is_root = self.is_root)
        temp_category.add_subcategories(not_cutoff_categories)
        temp_category.compute_grid(configs.width, configs.height)


        # recursively generate sub-treemaps for each subcategory
        if len(configs.styles) <= 1:  # if running out of user-specified styles, default to uniform
            configs.styles.append("uniform")

        color_legend = {}
        for i, subcategory in enumerate(not_cutoff_categories):
            # get new color for the subcategory
            sub_grid = temp_category.grids[subcategory.name]
            sub_color = color_style.get_color(category=subcategory, i=i, palette=configs.palette)
            color_legend[subcategory.name] = sub_color

            # create the new configs for the subcategory
            sub_configs = VisConfig(width = int(sub_grid["width"]),
                                    height = int(sub_grid["height"]),
                                    font = configs.font,
                                    base_font_size = configs.base_font_size,
                                    line_width = (configs.line_width * 2) // 3,
                                    proportion_cutoff = configs.proportion_cutoff,
                                    top_color = sub_color,
                                    line_color = configs.line_color,
                                    palette = configs.palette,
                                    styles = configs.styles[1:],
                                    label_generator = configs.label_generator,
                                    min_font_size = configs.min_font_size)
            sub_image = subcategory.visualize_main_map(sub_configs)["image"]
            img.paste(sub_image, (sub_grid["ul"]))
        

        ImageDraw.Draw(img).rectangle([(0,0), (configs.width, configs.height)],
                                    outline = configs.line_color,
                                    width = configs.line_width)

        return {"image":img, "color_legend":color_legend}
    
    def compute_simple_column_grid(self, width, height, categories, ul = (0,0)):
        categories.sort(key = lambda cat: cat.proportion_in_supcategory, reverse = True)
        total_column_size = sum([cat.size for cat in categories])
        current_ul = ul
        for cat in categories:
            height_used = int(height * (cat.size / total_column_size))
            current_grid = {"ul":current_ul, "width": width, "height":height_used}
            current_ul = (current_ul[0], current_ul[1] + height_used)
            self.grids[cat.name] = current_grid

    def compute_simple_row_grid(self, width, height, categories, ul = (0,0)):
        categories.sort(key = lambda cat: cat.proportion_in_supcategory, reverse = True)
        total_column_size = sum([cat.size for cat in categories])
        current_ul = ul
        for cat in categories:
            width_used = int(width * (cat.size / total_column_size))
            current_grid = {"ul":current_ul, "width": width_used, "height":height}
            current_ul = (current_ul[0] + width_used, current_ul[1])
            self.grids[cat.name] = current_grid

    def compute_grid(self, width, height, ul = (0,0), current_category_id=0):

        current_category = self.subcategories_in_descending_size[current_category_id]
        if current_category_id == len(self.subcategories) - 1:
            self.grids[current_category.name] = {"ul":ul, "width": width, "height":height}
            return
        
        total_size_remaining = sum([cat.size for cat in self.subcategories_in_descending_size[current_category_id:]])
        gridded_categories = [current_category]
        if width > height:
            current_width_required = int(width * current_category.size / total_size_remaining)
            first_category_height = height
            while (current_width_required < first_category_height) and (current_category_id < len(self.subcategories) - 1): # while there are more subcategories to be added...
                current_category_id += 1
                gridded_categories.append(self.subcategories_in_descending_size[current_category_id])
                current_width_required = int(width * sum([category.size for category in gridded_categories]) / total_size_remaining)
                first_category_height = height * gridded_categories[0].size / sum([category.size for category in gridded_categories])

            self.compute_simple_column_grid(width=current_width_required, height=height, ul=ul, categories=gridded_categories)
            if not current_category_id == len(self.subcategories) - 1:
                new_ul = (ul[0] + current_width_required, ul[1])
                self.compute_grid(width= width - current_width_required, height=height, ul=new_ul, current_category_id=current_category_id+1)
        else:
            current_height_required = int(height * current_category.size / total_size_remaining)
            first_category_width = width
            while (current_height_required < first_category_width) and (current_category_id < len(self.subcategories) - 1): # while there are more subcategories to be added...
                current_category_id += 1
                gridded_categories.append(self.subcategories_in_descending_size[current_category_id])
                current_height_required = int(height * sum([category.size for category in gridded_categories]) / total_size_remaining)
                first_category_width = width * gridded_categories[0].size / sum([category.size for category in gridded_categories])

            self.compute_simple_column_grid(width=width, height=current_height_required, ul=ul, categories=gridded_categories)
            if not current_category_id == len(self.subcategories) - 1:
                new_ul = (ul[0], current_height_required + ul[1])
                self.compute_grid(width=width, height=height - current_height_required, ul=new_ul, current_category_id=current_category_id+1)


    @property
    def label(self):
        if self.supcategory is None:
            return self.name
        if self.supcategory.is_root:
            return self.name
        return self.supcategory.label + " -> " + self.name
    
    @property
    def proportion_in_supcategory(self):
        return self.size  / self.supcategory.size
    
    @property
    def relative_proportion_in_supcategory(self):
        largest_sibling = self.supcategory.subcategories_in_descending_size[0]
        return self.size / largest_sibling.size
    
    @property
    def proportion_in_smaller_siblings(self):
        self_id = [i for i,cat in enumerate(self.supcategory.subcategories_in_descending_size) if cat.name == self.name][0]
        lower_siblings = self.supcategory.subcategories_in_descending_size[self_id:]
        return self.size / sum([cat.size for cat in lower_siblings])
    
    def subcategory_by_name(self, name):
        search_result = [cat for cat in self.subcategories if cat.name == name]
        if search_result:
            return search_result[0]
        raise ValueError("no matching subcategory found")
    
    def describe(self, n_indents=0, depth=-1, indent="\t", out=True) -> str:
        indent_string = f"\n{(n_indents + 1) * indent}"
        description = self.__str__()
        if not depth == 0:
            subcategories_described = [indent_string + subcategory.describe(n_indents = n_indents + 1, out=False, depth=depth-1) for subcategory in self.subcategories]
            description += "".join(subcategories_described)
        if out:
            print(description)
        else:
            return description

    def __str__(self) -> str:
        return f"Category [{self.label}] of size {round(self.size, 4)}"

class Legend:
    def __init__(self, color_dict):
        self.color_dict = color_dict
        self.category_boxes = {}
        self.wrapped_texts = {}
    
    @property
    def n_categories(self):
        return len(self.color_dict)

    def n_row_cols(self, configs):
        # number of rows or columns needed for the legend to list all categories
        return ceil(self.n_categories / configs.legend_rowcol_length)

    def categories_by_rowcol(self, configs):
        # a list of lists, where each sublist contains the names of categories belonging to a row/column of the legend
        categories = list(self.color_dict.keys())
        n_lists = self.n_row_cols(configs)
        return [categories[i:i+configs.legend_rowcol_length] for i in range(0, self.n_categories, configs.legend_rowcol_length)]

    def compute_category_boxes(self, configs):
        if configs.is_row_legend:
            useable_width_per_cat = (configs.width // min(configs.n_categories, configs.legend_rowcol_length))
        else:
            useable_width_per_cat = (configs.legend_rowcol_size)
        return self.compute_category_boxes_(configs, useable_width_per_cat)

    def compute_category_boxes_ (self, configs, useable_width):
        # finds the appropriate line-breaks for textual labels, then compute the size of the box of the category in the legend
        font = ImageFont.FreeTypeFont(font=configs.font, size=configs.legend_font_size)
        text_useable_width = useable_width - 2 * configs.legend_color_block_size
        def text_box(text):
            rows = textwrap.wrap(text, text_useable_width // font.getsize("H")[0])
            total_height = sum([font.getsize(row)[1] for row in rows])
            total_width = max([font.getsize(row)[0] for row in rows])
            return (rows, total_width, total_height)
        text_boxes = {cat_name: text_box(cat_name) for cat_name in self.color_dict.keys()}  # box for text, without the color code
        category_boxes = {cat_name: (width + 2 * configs.legend_color_block_size, height) for cat_name, (_, width, height) in text_boxes.items()}
        wrapped_texts = {cat_name: wrapped_text for cat_name, (wrapped_text, _, _) in text_boxes.items()}
        self.category_boxes = self.category_boxes | category_boxes
        self.wrapped_texts = self.wrapped_texts | wrapped_texts
        
    def compute_simple_column_grid(self, category_boxes, configs, ul=(0,0)):
        # compute a grid for arranging the categories in category_boxes into a single column
        total_content_height = sum([box[1] for box in category_boxes.values()])
        between_category_margin = (configs.height - total_content_height) // (len(category_boxes) + 2)
        configs.legend_margins.append(between_category_margin)

        left_margin = min([(configs.legend_rowcol_size - cat_width) // 2 for (cat_width, _) in category_boxes.values()])
        ul = (ul[0] + left_margin, ul[1] + between_category_margin)

        column_grids = {}

        for cat_name, (cat_width, cat_height) in category_boxes.items():
            cat_grid = {"ul":ul, "width":cat_width, "height":cat_height}
            column_grids[cat_name] = cat_grid
            ul = (ul[0], ul[1] + between_category_margin + cat_height)
        return column_grids
    
    def compute_simple_row_grid(self, category_boxes, configs, ul=(0,0)):
        # compute a grid for arranging the categories in category_boxes into a single row
        total_content_width = sum([box[0] for box in category_boxes.values()])
        between_category_margin = (configs.width - total_content_width) // (len(category_boxes) + 1)
        configs.legend_margins.append(between_category_margin)

        ul = (ul[0] + between_category_margin, ul[1])

        row_grids = {}

        for cat_name, (cat_width, cat_height) in category_boxes.items():
            cat_ul = (ul[0], ul[1] + (configs.legend_rowcol_size - cat_height) // 2)
            cat_grid = {"ul":cat_ul, "width":cat_width, "height":cat_height}
            row_grids[cat_name] = cat_grid
            ul = (ul[0] + between_category_margin + cat_width, ul[1])
        return row_grids
    
    def visualize_one_rowcol_legend(self, category_boxes, configs, ul=(0,0)):
        # draws a one-column legend containing the entries in category_boxes
        
        if configs.is_row_legend:
            rowcol_grid = self.compute_simple_row_grid(category_boxes, configs, ul=(ul[0], ul[1]))
            legend_img = Image.new('RGBA', (configs.width, configs.legend_rowcol_size), color = (0,0,0,0))
        else:
            rowcol_grid = self.compute_simple_column_grid(category_boxes, configs, ul=(ul[0], ul[1]))
            legend_img = Image.new('RGBA', (configs.legend_rowcol_size, configs.height), color = (0,0,0,0))

        font = ImageFont.FreeTypeFont(font=configs.font, size=configs.legend_font_size)

        for category, grid in rowcol_grid.items():
            box = category_boxes[category]
            color = self.color_dict[category]
            category_img = Image.new('RGBA', box, color = (0,0,0,0))
            color_block_upbottom_margin = (box[1] - configs.legend_color_block_size) // 2
            ImageDraw.Draw(category_img).rectangle([(0,color_block_upbottom_margin),
                                                    (configs.legend_color_block_size, color_block_upbottom_margin + configs.legend_color_block_size)],
                                                    outline = color, fill = color)
            category_wrapped_text = "\n".join(self.wrapped_texts[category])
            ImageDraw.Draw(category_img).text((configs.legend_color_block_size * 2, grid["height"] // 2),
                                              category_wrapped_text, fill=color, anchor='lm', font=font)
            legend_img.paste(category_img, grid["ul"], mask=category_img)
        
        return legend_img
    

    def paste_rowcols(self, configs, rowcols):
        if configs.is_row_legend:
            return self.paste_rows(configs, rows=rowcols)
        else:
            return self.paste_cols(configs, cols=rowcols)

    def paste_cols(self, configs, cols):
        # paste images of columns together, stacked horizontally
        max_height = max([col.size[1] for col in cols])
        total_width = sum([col.size[0] for col in cols]) + configs.legend_rowcol_margin * len(cols)
        pasted_img = Image.new('RGB', (total_width, max_height), color = configs.top_color)
        ul = (0,0)
        for col in cols:
            pasted_img.paste(col, ul, mask=col)
            ul = (ul[0] + col.size[0] + configs.legend_rowcol_margin, ul[1])
        return pasted_img
    
    def paste_rows(self, configs, rows):
        # paste images of rows together, stacked vertically
        total_height = sum([row.size[1] for row in rows]) + configs.legend_rowcol_margin * (1 + len(rows))
        max_width = max([row.size[0] for row in rows]) 
        pasted_img = Image.new('RGB', (max_width, total_height), color = configs.top_color)
        ul = (0, configs.legend_rowcol_margin)
        for row in rows:
            pasted_img.paste(row, ul, mask=row)
            ul = (ul[0], ul[1] + row.size[1] + 2 * configs.legend_rowcol_margin)
        return pasted_img


    def visualize(self, configs):
        categories_by_rowcol = self.categories_by_rowcol(configs)
        self.compute_category_boxes(configs)
        category_boxes_by_rowcol = [{cat:self.category_boxes[cat] for cat in row} for row in categories_by_rowcol]

        rowcol_imgs = []

        for boxes_in_rowcol in category_boxes_by_rowcol:
            rowcol_img = self.visualize_one_rowcol_legend(category_boxes = boxes_in_rowcol, configs=configs)
            rowcol_imgs.append(rowcol_img)

        return self.paste_rowcols(configs, rowcol_imgs)
        
            


from dataclasses import dataclass

@dataclass
class VisConfig:
    '''class for storing configurations for visualizing the tree map'''
    width: int
    height: int
    font: str = "NotoSans-Regular.ttf"
    base_font_size: int = 20
    min_font_size: int = 15
    text_color: int = "black"
    text_level: int | str = "all"
    line_width: int = 10
    line_color: str = "black"
    proportion_cutoff: int | None = None
    n_categories: int | None = None
    top_color: str = "white"
    palette: list | None = None # a list containing hex color codes for each top-level category
    styles: list | None = None  # a list containing entries "palette", "gradient", or "uniform", length equal to the depth of the tree map
    legend: bool = True
    legend_position: str = "bottom"
    legend_font_size: int | None = None
    legend_color_block_size: int | None = None
    legend_margins = []
    legend_rowcol_size: int | None = None
    legend_rowcol_length: int = 5  # number of categories per row/column of the legend
    legend_rowcol_margin: int = 0   # size of extra margin between legend row/columns
    label_generator: callable = lambda x: x

    def __post_init__(self):
        if self.palette is None:
            if self.n_categories is not None:
                self.palette = list(sns.color_palette("pastel", self.n_categories).as_hex())
            else:
                raise ValueError("Please provide a value for at least one of palette and n_categories")
        elif self.n_categories is None:
            self.n_categories = len(self.palette)
        elif not self.n_categories == len(self.palette):
            raise ValueError(f"n_categories ({self.n_categories}) should eqaul len(palette), which is {len(self.palette)}")
        
        if self.styles is None:
            self.styles = ["palette", "gradient", "uniform"]
        
        if self.legend_position not in ["left", "right", "top", "bottom"]:
            raise ValueError(f"{self.legend_position} is not a valid position.")
        
        if self.legend_rowcol_size is None:
            self.legend_rowcol_size = (self.width + self.height) // 8
        
        if self.legend_color_block_size is None:
            self.legend_color_block_size = min(self.width, self.height) // 50

        if self.legend_font_size is None:
            self.legend_font_size = self.base_font_size

    @property
    def is_row_legend(self):    
        return self.legend_position in ["top", "bottom"]

    @property
    def top_level_color_style(self):
        return ColorStyle(style = self.styles[0], base_color = self.top_color)

    def total_size(self, legend_size):
        if self.is_row_legend:
            return (self.width, self.height + legend_size[1])
        else:
            return (self.width + legend_size[0], self.height)
    
    @property
    def legend_ul(self):
        # the upper-left corner coordinates of the legend
        match self.legend_position:
            case "top":
                return (0,0)
            case "bottom":
                return (0, self.height)
            case "left":
                return (0,0)
            case "right":
                return (self.width, 0)
    
    @property
    def main_map_ul(self):
        match self.legend_position:
            case "top":
                return (0, self.legend_height)
            case "bottom":
                return (0, 0)
            case "left":
                return (self.legend_width, 0)
            case "right":
                return (0, 0)

@dataclass
class ColorStyle:
    style: str
    base_color: str | Color = ""
    others_color: str | Color | None = None
    
    def __post_init__(self):
        if self.style not in ["palette", "gradient", "uniform"]:
            raise ValueError("Invalid style")

        if isinstance(self.base_color, str):
            if self.base_color == "":
                self.base_color = Color("white")
            else:
                self.base_color = Color(self.base_color)
        
        if isinstance(self.others_color, str):
            self.others_color = Color(self.others_color)
    
    def get_others_color(self, palette):
        if self.others_color is None:
            palette_avg_luminance = sum([Color(hex).luminance for hex in palette]) / len(palette)
            others_color = Color("white")
            others_color.luminance = palette_avg_luminance
            return others_color.hex
        return self.others_color.hex

    def get_color(self, **kwargs):
        match self.style:
            case "palette":
                is_other_category = kwargs["category"].is_others
                palette = kwargs["palette"]
                if is_other_category:
                    return self.get_others_color(palette)
                id = kwargs["i"]
                return palette[id]
            
            case "gradient":
                category = kwargs["category"]
                scaled_color = deepcopy(self.base_color)
                scaled_color.saturation *= category.relative_proportion_in_supcategory
                return scaled_color.hex
            
            case "uniform":
                return self.base_color
    
    @property
    def base_hex(self):
        return self.base_color.hex
    
class StringHelper:
    def split_lines(text, configs, font_func):
        '''helper function for splitting the label of this category into lines.
        [font_func] takes an integer representing the font size, returns a font of that size.'''
        lines = []
        words = text.split(" ")
        word_widths = deque([(word, font_func(configs.base_font_size).getsize(word)[0]) for word in words])
        new_line = ""
        new_line_length = 0
        while word_widths:
            new_word, new_word_width = word_widths.popleft()
            new_line += new_word + " "
            new_line_length += new_word_width
            if new_line_length > configs.width:
                lines.append(new_line.strip())
                new_line = ""
                new_line_length = 0
        if new_line:
            lines.append(new_line.strip())
        longest_line = max(lines, key = lambda line: font_func(10).getsize(line)[0])
        return ("\n".join(lines), longest_line)

    def largest_fitting_font_size(text, font_func, max_width, max_height, max_font_size=None, min_font_size=15):
        '''Helper function that finds the largest font size that [text] can use to fit into a box of [max_width]*[max_height].
        [min_font_size] specifies a cutoff -- if the text needs a font size smaller than that, it is not displayed at all (with size 0)
        [max_font_size] caps the font size.'''

        font_size = 1
        text_width = lambda font_size: font_func(font_size).getsize(text)[0]
        text_height = lambda font_size: font_func(font_size).getsize(text)[1]

        while (text_width(font_size) < max_width) and (text_height(font_size) < max_height):
            if max_font_size is not None and font_size > max_font_size:
                break
            font_size += 1
        if (font_size - 1) >= min_font_size:
            return (font_size - 1)
        return 0