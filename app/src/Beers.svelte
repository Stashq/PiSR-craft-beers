<script lang="ts">
	export let name: string;
	import Menu from '@smui/menu';
	import List, {Separator, Text } from '@smui/list';
	import Button from '@smui/button';
	import BeerList from './Beer_image_list.svelte'
	import ImageList, {
			  Item,
			  ImageAspectContainer,
			  Image,
			  Supporting,
			  Label,
			} from '@smui/image-list';
	
</script>

	<div style="min-width: 100px; width:100%">
			{#each Array(3) as _unused, i}
			  <BeerList></BeerList>
			{/each}
	  </div>

<style>
	.beer_img{
		width: 50%;
		height: 70%;
	}
</style>